"""
Module 7 Week B — Applied Lab: Pre-Trained QA Evaluation on Tech/Entertainment News.

Implement the functions below. See the lab guide for full task descriptions.
"""

import json
import os
import re
import string
from collections import Counter

import pandas as pd


# -- Helpers (provided — do NOT modify) --------------------------------------

def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI smoke fixture), otherwise the default
    path to the curated tech/entertainment QA CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/tech_news_qa.csv")


def get_qa_model_name() -> str:
    """Return the QA model name. Provided helper. Do not modify."""
    return "distilbert-base-cased-distilled-squad"


def load_examples(data_path: str) -> pd.DataFrame:
    """
    Load the curated QA CSV.

    Returns a DataFrame with columns: qid, question, context, gold_answer.

    The default `data/tech_news_qa.csv` ships ~1,000 extractive QA tuples
    derived from glnmario/news-qa-summarization (CNN tech / entertainment
    slice). Each gold answer is a literal substring of its context, by
    construction (filtered during curation).

    Provided helper. Do not modify.
    """
    df = pd.read_csv(data_path)
    required = {"qid", "question", "context", "gold_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")
    return df


# -- Task 1: Normalization + EM + F1 (same as drill) -------------------------

def normalize_answer(s: str) -> str:
    """SQuAD-style normalization (see drill / reading)."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> int:
    """Return 1 if normalized prediction equals normalized gold."""
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """
    Token-F1 between prediction and gold after normalization.

    Empty handling:
      - both empty -> 1.0
      - one empty -> 0.0
    Returns float in [0.0, 1.0]; never NaN.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# -- Task 2: Build the QA pipeline -------------------------------------------

def build_qa_pipeline(model_name: str):
    """Construct a Hugging Face question-answering pipeline."""
    from transformers import pipeline

    return pipeline("question-answering", model=model_name, tokenizer=model_name)


# -- Task 3: Predict one answer ---------------------------------------------

def predict_one(qa, question: str, context: str) -> str:
    """
    Run the QA pipeline on one (question, context) pair.

    Returns the answer STRING only (not the full pipeline output dict).
    """
    result = qa(question=str(question), context=str(context))
    return str(result.get("answer", ""))


# -- Task 4: Evaluate over the dataset ---------------------------------------

def evaluate_qa(qa, examples: pd.DataFrame) -> dict:
    """
    Evaluate the QA pipeline over a DataFrame of examples.

    Returns:
        {
          "em": float,   # mean EM
          "f1": float,   # mean token-F1
          "n": int,
          "predictions": [
            {qid, question, context_excerpt, gold_answer, predicted_answer, em, f1},
            ...
          ],
        }
    context_excerpt is the first 80 chars of the context (CSV-friendly).
    """
    predictions = []
    em_scores = []
    f1_scores = []

    for _, row in examples.iterrows():
        question = str(row["question"])
        context = str(row["context"])
        gold = str(row["gold_answer"])
        predicted = predict_one(qa, question, context)
        em = exact_match(predicted, gold)
        f1 = token_f1(predicted, gold)

        em_scores.append(em)
        f1_scores.append(f1)
        predictions.append(
            {
                "qid": row["qid"],
                "question": question,
                "context_excerpt": context[:80],
                "gold_answer": gold,
                "predicted_answer": predicted,
                "em": em,
                "f1": f1,
            }
        )

    n = len(predictions)
    return {
        "em": float(sum(em_scores) / n) if n else 0.0,
        "f1": float(sum(f1_scores) / n) if n else 0.0,
        "n": n,
        "predictions": predictions,
    }


# -- Task 5: Orchestrate -----------------------------------------------------

def main() -> None:
    """Load data, build pipeline, evaluate, write artifacts."""
    examples = load_examples(get_data_path())
    qa = build_qa_pipeline(get_qa_model_name())
    result = evaluate_qa(qa, examples)

    # Write predictions CSV
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv("qa_predictions.csv", index=False)

    # Write metrics JSON
    metrics = {
        "em": result["em"],
        "f1": result["f1"],
        "n": result["n"],
        "model": get_qa_model_name(),
    }
    with open("qa_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Aggregate EM = {result['em']:.4f}")
    print(f"Aggregate F1 = {result['f1']:.4f}")
    print(f"n = {result['n']}")


if __name__ == "__main__":
    main()
