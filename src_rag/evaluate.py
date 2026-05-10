from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep

from src_rag import models

FOLDER = Path("td8") / "wiki"
FILENAMES = [
    FOLDER / title
    for title in [
        "Inception.md",
        "The Dark Knight.md",
        "Deadpool.md",
        "Fight Club.md",
        "Pulp Fiction.md",
    ]
]

_DF = None
_ENCODER = None


def _get_df():
    global _DF
    if _DF is None:
        _DF = pd.read_csv("td8/questions.csv", sep=";")
    return _DF


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


def _load_mlflow():
    conf = models._get_conf()
    experiment = conf.get("mlflow_experiment", "RAG_Movies")
    mlflow.set_experiment(experiment)


# ---------------------------------------------------------------------------
# Public entry points (called by the user / notebook)
# ---------------------------------------------------------------------------

def run_evaluate_retrieval(config: dict, rag=None):
    """
    Evaluate retrieval quality (MRR) and log results to MLflow.

    Example:
        from src_rag import evaluate
        evaluate.run_evaluate_retrieval(config={})
    """
    _load_mlflow()
    rag = rag or models.get_model(config)
    df = _get_df().dropna()
    score = evaluate_retrieval(rag, FILENAMES, df)

    description = str(config.get("model", "default"))
    _push_mlflow_result(score, config, description)
    return rag


def run_evaluate_reply(config: dict, rag=None):
    """
    Evaluate answer quality (semantic similarity) and log to MLflow.
    Uses every 10th question to limit Groq API calls.
    """
    _load_mlflow()
    rag = rag or models.get_model(config)
    df = _get_df()
    indexes = range(2, len(df), 10)
    score = evaluate_reply(rag, FILENAMES, df.iloc[indexes])

    description = str(config.get("model", "default"))
    _push_mlflow_result(score, config, description)
    return rag


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def evaluate_retrieval(rag, filenames, df_question: pd.DataFrame) -> dict:
    """
    For each question, check whether the expected text is in the top-5
    retrieved chunks and compute MRR (Mean Reciprocal Rank).
    """
    rag.load_files(filenames)
    ranks = []

    for _, row in df_question.iterrows():
        chunks = rag._get_context(row.question)
        try:
            rank = 1 + next(
                i for i, c in enumerate(reversed(chunks))
                if row.text_answering in c
            )
        except StopIteration:
            rank = 0

        ranks.append(rank)

    df_question = df_question.copy()
    df_question["rank"] = ranks
    mrr = np.mean([0 if r == 0 else 1 / r for r in ranks])

    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }


def evaluate_reply(rag, filenames, df: pd.DataFrame) -> dict:
    """
    Generate replies for each question and measure semantic similarity
    with the expected answer.
    """
    rag.load_files(filenames)
    replies = []

    for question in tqdm(df["question"], desc="Generating replies"):
        replies.append(rag.reply(question))
        sleep(2)  # Avoid hitting Groq rate limits

    df = df.copy()
    df["reply"] = replies
    df["sim"] = df.apply(
        lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]),
        axis=1,
    )
    df["is_correct"] = df["sim"] > 0.7

    return {
        "reply_similarity": df["sim"].mean(),
        "percent_correct": df["is_correct"].mean(),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push_mlflow_result(score: dict, config: dict, description: str = None):
    with mlflow.start_run(description=description):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics({k: v for k, v in score.items() if isinstance(v, (int, float))})

        config_no_key = {k: v for k, v in config.items() if not k.endswith("_key")}
        mlflow.log_dict(config_no_key, "config.json")


def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    """Cosine similarity between two texts using sentence-transformers."""
    encoder = _get_encoder()
    embeddings = encoder.encode([generated_answer, reference_answer])
    gen_emb = embeddings[0].reshape(1, -1)
    ref_emb = embeddings[1].reshape(1, -1)
    return float(cosine_similarity(gen_emb, ref_emb)[0][0])


def calc_acceptable_chunks(chunks: list[str], text_to_find: list[str]) -> list[set]:
    return [
        {i for i, chunk in enumerate(chunks) if answer in chunk}
        for answer in text_to_find
    ]


def calc_mrr(sim_score, acceptable_chunks, top_n: int = 5) -> dict:
    ranks = []
    for this_score, this_acceptable in zip(sim_score, acceptable_chunks):
        indexes = reversed(np.argsort(this_score))
        try:
            rank = 1 + next(i for i, idx in enumerate(indexes) if idx in this_acceptable)
        except StopIteration:
            rank = len(this_score) + 1
        ranks.append(rank)

    return {
        "mrr": sum(1 / r if r <= top_n else 0 for r in ranks) / len(ranks),
        "ranks": ranks,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configs_to_test = [
        # Baseline
        {"model": {"chunk_size": 256, "overlap": 0, "use_small2big": False}},
        # Overlap chunking
        {"model": {"chunk_size": 256, "overlap": 64, "use_small2big": False}},
        # Small2Big
        {"model": {"chunk_size": 64, "overlap": 0, "use_small2big": True, "big_chunk_size": 256}},
        # Small chunk size
        {"model": {"chunk_size": 128, "overlap": 0, "use_small2big": False}},
        # Large chunk size
        {"model": {"chunk_size": 512, "overlap": 0, "use_small2big": False}},
    ]

    for cfg in configs_to_test:
        print(f"\n=== Testing config: {cfg['model']} ===")
        run_evaluate_retrieval(cfg)
