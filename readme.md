# RAGE — RAG-based Assistant for Graduate Employment

> **Longa** — Assistant IA de Loïc NGASSA, conçu pour répondre aux questions des recruteurs à partir de ses documents personnels.

---

## Présentation

**RAGE** est un chatbot RAG modulaire. Il exploite une base de données vectorielle FAISS construite à partir des documents PDF de Loïc NGASSA, et génère des réponses précises via le LLM **Llama 3.1 8B Instruct** hébergé sur HuggingFace.

---

## Architecture

```
data/                        ← Documents PDF sources
    ↓  ingest.py
faiss_index/                 ← Base vectorielle persistante 
    ↓  src/backend.py
Llama-3.1-8B (featherless)   ← LLM de génération via HuggingFace Inference API
    ↓  src/ui.py
Interface Streamlit           ← Chat interactif avec affichage des sources
```

### Flux RAG complet

```
Question utilisateur
      ↓
[all-mpnet-base-v2] → encodage en vecteur 768d
      ↓
[FAISS + MMR] → k=6 chunks pertinents et diversifiés (fetch_k=12)
      ↓
[Prompt Longa] → contexte injecté + règles strictes anti-hallucination
      ↓
[Llama-3.1-8B-Instruct] → réponse générée (max 512 tokens)
      ↓
Affichage + sources
```

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Interface | Streamlit 1.63+ |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` via featherless-ai |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768d) |
| Base vectorielle | FAISS CPU |
| Orchestration | LangChain (langchain-huggingface, langchain-community) |
| Évaluation | RAGAS 0.2.x |
| Gestion des dépendances | `uv` (Python 3.13) |
| Chunking | `RecursiveCharacterTextSplitter` — chunk 1000, overlap 150 |

---

## Structure du projet

```
RAGE/
├── data/                        # PDFs sources (non versionnés)
├── faiss_index/                 # Index vectoriel généré
│   ├── index.faiss
│   └── index.pkl
├── src/
│   ├── __init__.py
│   ├── backend.py               # Chargement modèles, chaîne RAG
│   └── ui.py                    # Interface Streamlit
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py              # Pipeline d'évaluation RAGAS
├── eval_results/                # Rapports générés après évaluation
│   ├── eval_detail_*.csv
│   └── eval_summary_*.json
├── golden_dataset.json          # 11 questions de référence pour l'évaluation
├── ingest.py                    # Pipeline d'ingestion des PDFs → FAISS
├── app.py                       # Point d'entrée Streamlit
├── pyproject.toml
└── .env                         
```

---

## Installation

### Prérequis

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) installé
- Un token HuggingFace avec accès au modèle `meta-llama/Llama-3.1-8B-Instruct`

### Mise en place

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd RAGE

# 2. Installer les dépendances
uv sync

# 3. Configurer les variables d'environnement
cp .env.example .env
# Renseigner HUGGINGFACEHUB_API_TOKEN dans .env
```

---

## Utilisation

### 1. Ingestion des documents

Placer les PDFs dans le dossier `data/`, puis lancer :

```bash
uv run python ingest.py
```

> À effectuer **une seule fois**, ou après ajout/modification de documents.  
> L'index FAISS est ensuite persisté sur disque et ne nécessite plus de ré-ingestion.

### 2. Lancer l'assistant

```bash
uv run streamlit run app.py
```

L'interface est accessible sur `http://localhost:8501`.

---

## Évaluation avec RAGAS

Le pipeline d'évaluation mesure la qualité du système RAG sur un ensemble de 11 questions de référence couvrant différents types de requêtes recruteur.

```bash
uv run python -m evaluation.evaluate \
    --dataset golden_dataset.json \
    --faiss-index ./faiss_index
```

Les résultats sont sauvegardés dans `eval_results/`.

### Métriques évaluées

| Métrique | Description |
|----------|-------------|
| `faithfulness` | La réponse est-elle fidèle au contexte récupéré ? |
| `answer_relevancy` | La réponse répond-elle à la question posée ? |
| `context_precision` | Les passages récupérés sont-ils pertinents ? |
| `context_recall` | Le contexte couvre-t-il ce qui est nécessaire pour répondre ? |

### Résultats de référence dernier run du 17/09/2026

| Métrique | Score |
|----------|:-----:|
| faithfulness | **0.81** |
| answer_relevancy | **0.83** |
| context_precision | **0.67** |
| context_recall | **0.76** |

> Le LLM juge utilisé pour l'évaluation est identique au LLM de génération (Llama-3.1-8B via featherless-ai).

---

## Choix de conception

### Séparation ingestion / runtime

L'ingestion (`ingest.py`) est un **pipeline offline** distinct de l'application. L'index FAISS est persisté sur disque et chargé au démarrage, les documents ne sont jamais ré-encodés à l'exécution.

### Chargement du modèle d'embedding au runtime

Bien que l'index FAISS soit pré-calculé, le modèle d'embedding doit rester **en mémoire au runtime** pour encoder les questions entrantes avant chaque recherche. Le décorateur `@st.cache_resource` garantit qu'il n'est chargé qu'une seule fois par session Streamlit.

### Recherche MMR

La recherche **Maximal Marginal Relevance** sélectionne des chunks à la fois pertinents et **diversifiés**, évitant la redondance lorsque plusieurs passages similaires existent dans les documents.

### Prompt anti-hallucination

Le prompt système interdit explicitement tout vocabulaire spéculatif ("probablement", "il semble", etc.) et impose une réponse standard lorsqu'une information est absente des documents.

---

## Limitations connues

- **Mémoire RAM** : le modèle d'embedding (~420 MB) et l'index FAISS cohabitent en mémoire. L'évaluation et l'app ne doivent pas tourner simultanément sur des machines à mémoire limitée.
- **Langue** : l'assistant répond en français par défaut et s'adapte à la langue du recruteur.
- **Périmètre** : limité aux informations présentes dans les documents fournis, toute question hors périmètre reçoit une réponse d'absence d'information.