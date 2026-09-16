"""
Pipeline d'évaluation pour RAGE (RAG QA sur PDF).

Ce script :
1. Charge le golden dataset
2. Fait tourner chaque question à travers la chaîne RAG existante
3. Calcule les métriques RAGAS : faithfulness, answer_relevancy,
   context_precision, context_recall
4. Sauvegarde les résultats détaillés par question + un résumé agrégé
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Chargement de la chaîne RAG (refactor de la logique de app.py)
# ---------------------------------------------------------------------------

def load_rag_chain(faiss_index_path: str, k: int = 3, repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Recharge la chaîne RAG à partir d'un index FAISS déjà construit.

    Séparé de la construction de l'index pour pouvoir évaluer sans
    re-vectoriser à chaque run.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.chains import RetrievalQA

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
    )
    chat_llm = ChatHuggingFace(llm=llm)

    prompt_template = """<s>[INST] Utilise le contexte suivant pour répondre à la question. Si tu ne sais pas, dis-le simplement.

Contexte : {context}

Question : {question} [/INST]</s>"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# ---------------------------------------------------------------------------
# 2. Exécution du golden dataset à travers la chaîne
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    question_id: str
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list = field(default_factory=list)
    error: str = ""


def run_dataset(chain, dataset_path: str) -> list[EvalRow]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[EvalRow] = []
    examples = data["examples"]
    print(f"Exécution de {len(examples)} questions sur la chaîne RAG...")

    for i, ex in enumerate(examples, 1):
        row = EvalRow(
            question_id=ex["id"],
            question=ex["question"],
            ground_truth=ex["ground_truth"],
        )
        try:
            result = chain.invoke({"query": ex["question"]})
            row.answer = result["result"]
            row.contexts = [doc.page_content for doc in result["source_documents"]]
        except Exception as e:
            row.error = str(e)
            print(f"  [{i}/{len(examples)}] ERREUR sur {ex['id']}: {e}")
            continue
        print(f"  [{i}/{len(examples)}] OK: {ex['id']}")
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# 3. Évaluation RAGAS
# ---------------------------------------------------------------------------

def evaluate_with_ragas(rows: list[EvalRow]) -> pd.DataFrame:
    """Calcule faithfulness, answer_relevancy, context_precision, context_recall.

    faithfulness        : la réponse est-elle fidèle au contexte récupéré (pas d'hallucination) ?
    answer_relevancy     : la réponse répond-elle vraiment à la question posée ?
    context_precision    : les passages récupérés sont-ils pertinents (peu de bruit) ?
    context_recall       : le contexte récupéré couvre-t-il ce qu'il faut pour répondre ?
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    eval_data = {
        "question": [r.question for r in rows],
        "answer": [r.answer for r in rows],
        "contexts": [r.contexts for r in rows],
        "ground_truth": [r.ground_truth for r in rows],
    }
    ds = Dataset.from_dict(eval_data)

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = result.to_pandas()
    df.insert(0, "question_id", [r.question_id for r in rows])
    return df


# ---------------------------------------------------------------------------
# 4. Rapport
# ---------------------------------------------------------------------------

def save_report(df: pd.DataFrame, output_dir: str, run_label: str = ""):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_label}" if run_label else ""

    detail_path = Path(output_dir) / f"eval_detail{suffix}_{timestamp}.csv"
    df.to_csv(detail_path, index=False)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = df[metric_cols].mean().to_dict()
    summary["n_questions"] = len(df)
    summary["run_label"] = run_label or "default"
    summary["timestamp"] = timestamp

    summary_path = Path(output_dir) / f"eval_summary{suffix}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== RÉSUMÉ ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nDétail par question : {detail_path}")
    print(f"Résumé agrégé       : {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG avec RAGAS")
    parser.add_argument("--dataset", default="golden_dataset.json")
    parser.add_argument("--faiss-index", default="./faiss_index")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("ERREUR : HUGGINGFACEHUB_API_TOKEN n'est pas défini (fichier .env).")
        sys.exit(1)

    chain = load_rag_chain(args.faiss_index, k=args.k)
    rows = run_dataset(chain, args.dataset)

    if not rows:
        print("Aucune ligne évaluable (toutes les questions ont échoué).")
        sys.exit(1)

    df = evaluate_with_ragas(rows)
    save_report(df, args.output_dir, run_label=args.run_label)


if __name__ == "__main__":
    main()
