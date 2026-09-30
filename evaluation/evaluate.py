"""
Pipeline d'évaluation pour Longa.

Ce script :
1. Charge le golden dataset
2. Fait tourner chaque question à travers la chaîne RAG existante
3. Calcule les métriques RAGAS : faithfulness, answer_relevancy,
   context_precision, context_recall
4. Sauvegarde les résultats détaillés par question + un résumé agrégé
"""

import argparse
import json
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

LLM_MODEL   = "openai/gpt-oss-120b" 
JUDGE_MODEL = "openai/gpt-oss-120b"      

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def make_llm(model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY manquante dans le .env")
    return ChatOpenAI(
        model=model,
        base_url=GROQ_BASE_URL,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=5,
    )


# 1. Chargement de la chaîne RAG
def load_rag_chain(faiss_index_path: str, k: int = 3):
    """Recharge la chaîne RAG à partir d'un index FAISS déjà construit.

    Séparé de la construction de l'index pour pouvoir évaluer sans
    re-vectoriser à chaque run.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.chains import RetrievalQA

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    chat_llm = make_llm(LLM_MODEL, temperature=0.2, max_tokens=1024)

    prompt_template = """Tu es Longa, l'assistant professionnel de Loïc NGASSA.
Ta mission est de répondre aux questions de recruteurs concernant
le parcours, les compétences, les expériences, les projets,
la formation et les réalisations de Loïc NGASSA.

RÈGLE PRINCIPALE :
Tu dois répondre uniquement à partir des informations présentes
dans le contexte documentaire fourni.
Tu ne dois jamais inventer, supposer, extrapoler ou déduire une
information personnelle concernant Loïc NGASSA.
Si une information n'est pas présente dans le contexte fourni,
dis-le explicitement.

N'utilise jamais des formulations telles que :
- "probablement"
- "on peut supposer"
- "il semble"
- "cela suggère"
- "il a probablement"

pour compléter une information absente.
Si la question porte sur une information absente des documents,
réponds :
"Je ne dispose pas de cette information dans les documents qui
m'ont été fournis et je préfère ne pas spéculer."

Lorsque plusieurs informations pertinentes sont disponibles,
synthétise-les clairement sans ajouter d'informations externes.

Pour les compétences, distingue :
- compétences explicitement mentionnées ;
- technologies explicitement utilisées ;
- expériences/projets ayant permis de développer ces compétences.

Ne transforme pas automatiquement une expérience en compétence
maîtrisée.
Lorsque c'est pertinent, indique le projet ou l'expérience
à l'origine de l'information.
Tu dois toujours privilégier la précision à la quantité.
Tu réponds en français sauf si le recruteur utilise une autre langue.

Contexte : {context}

Question : {question}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# 2. Exécution du golden dataset à travers la chaîne
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
    print(f"Exécution de {len(examples)} questions sur le pipeline...")

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


# 3. Évaluation RAGAS
def evaluate_with_ragas(rows: list[EvalRow]) -> pd.DataFrame:
    """Calcule faithfulness, answer_relevancy, context_precision, context_recall.

    faithfulness        : la réponse est-elle fidèle au contexte récupéré ?
    answer_relevancy    : la réponse répond-elle vraiment à la question posée ?
    context_precision   : les passages récupérés sont-ils pertinents (peu de bruit) ?
    context_recall      : le contexte récupéré couvre-t-il ce qu'il faut pour répondre ?
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_huggingface import HuggingFaceEmbeddings

    # LLM juge : température 0 et plus de tokens (RAGAS attend des sorties JSON)
    judge_llm = make_llm(JUDGE_MODEL, temperature=0.0, max_tokens=2048)
    ragas_llm = LangchainLLMWrapper(judge_llm)

    # Embeddings juge : même modèle que l'index FAISS
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # Groq n'accepte pas n > 1 : on force une seule question générée par réponse
    answer_relevancy.strictness = 1

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
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=6),  # évite les 429
    )
    df = result.to_pandas()
    df.insert(0, "question_id", [r.question_id for r in rows])
    return df


# 4. Rapport
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


# main
def main():
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG avec RAGAS")
    parser.add_argument("--dataset", default="golden_dataset.json")
    parser.add_argument("--faiss-index", default="./faiss_index")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()

    chain = load_rag_chain(args.faiss_index, k=args.k)
    rows = run_dataset(chain, args.dataset)

    if not rows:
        print("Aucune ligne évaluable (toutes les questions ont échoué).")
        sys.exit(1)

    df = evaluate_with_ragas(rows)
    save_report(df, args.output_dir, run_label=args.run_label)


if __name__ == "__main__":
    main()