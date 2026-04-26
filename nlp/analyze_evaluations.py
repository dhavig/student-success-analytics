"""
Analyze the synthetic course-evaluation corpus.

For each comment:
  - VADER compound sentiment score in [-1, 1].
  - LDA topic distribution over 6 latent topics.

Aggregates results to the student level and writes:
  artifacts/evaluation_sentiment.csv  per-student sentiment + ratings
  artifacts/evaluation_topics.csv     LDA topic-term distributions
  artifacts/student_topics.csv        per-student topic distribution
  artifacts/sentiment_vs_retention.csv  group-level summary

This is the "survey + unstructured text" component the JD calls out.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ART_DIR = os.path.join(ROOT, "artifacts")

N_TOPICS = 6
TOP_TERMS_PER_TOPIC = 8

# Topic labels are inferred manually after inspecting top terms; if LDA finds
# different clusters, the labels in the dashboard will still degrade gracefully.
SUGGESTED_LABELS = [
    "Instruction & engagement",
    "Workload & assessment",
    "Financial stress",
    "Advising & support",
    "Belonging & community",
    "Career relevance",
]

STOPWORDS = {
    "the", "and", "was", "were", "this", "that", "with", "for", "from",
    "but", "they", "them", "have", "had", "are", "you", "i", "me", "my",
    "we", "our", "their", "his", "her", "would", "could", "should",
    "of", "in", "on", "at", "to", "a", "an", "is", "be", "as", "it",
    "all", "any", "some", "more", "most", "not", "no", "than", "too",
    "during", "made", "make", "really", "very", "much", "even",
    "felt", "feel", "feeling", "found", "find",
    "course", "class", "semester", "professor", "instructor",
    "lecture", "lectures", "assignment", "assignments",
}


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]


def main() -> None:
    in_path = os.path.join(DATA_DIR, "course_evaluations.csv")
    if not os.path.exists(in_path):
        raise SystemExit(f"Missing {in_path}. Run `python etl/generate_evaluations.py` first.")

    os.makedirs(ART_DIR, exist_ok=True)
    evals = pd.read_csv(in_path)
    print(f"Loaded {len(evals):,} evaluations.")

    analyzer = SentimentIntensityAnalyzer()
    evals["sentiment"] = evals["comment"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    vectorizer = CountVectorizer(
        tokenizer=_tokenize,
        lowercase=False,
        max_df=0.85,
        min_df=5,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(evals["comment"])
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="online",
        max_iter=20,
    )
    doc_topic = lda.fit_transform(X)

    # Per-document dominant topic
    evals["dominant_topic"] = doc_topic.argmax(axis=1)
    for t in range(N_TOPICS):
        evals[f"topic_{t}"] = doc_topic[:, t]

    topic_term_rows = []
    for k in range(N_TOPICS):
        top_idx = lda.components_[k].argsort()[::-1][:TOP_TERMS_PER_TOPIC]
        terms = [(feature_names[i], float(lda.components_[k, i])) for i in top_idx]
        label = SUGGESTED_LABELS[k] if k < len(SUGGESTED_LABELS) else f"Topic {k+1}"
        topic_term_rows.append({
            "topic": k,
            "label": label,
            "top_terms": " | ".join(t for t, _ in terms),
            "term_weights": " | ".join(f"{w:.2f}" for _, w in terms),
        })
    pd.DataFrame(topic_term_rows).to_csv(
        os.path.join(ART_DIR, "evaluation_topics.csv"), index=False
    )

    # Per-student aggregation
    student_agg = (evals.groupby("student_id")
                        .agg(n_evals=("rating", "count"),
                             mean_rating=("rating", "mean"),
                             mean_sentiment=("sentiment", "mean"),
                             min_sentiment=("sentiment", "min"),
                             share_negative=("sentiment", lambda s: (s < -0.1).mean()))
                        .reset_index())
    topic_cols = [f"topic_{t}" for t in range(N_TOPICS)]
    student_topic = (evals.groupby("student_id")[topic_cols].mean().reset_index())
    student_agg = student_agg.merge(student_topic, on="student_id")
    student_agg.to_csv(os.path.join(ART_DIR, "evaluation_sentiment.csv"), index=False)

    student_topic.to_csv(os.path.join(ART_DIR, "student_topics.csv"), index=False)

    # Sentiment vs retention -- the killer cross-link
    retention_path = os.path.join(DATA_DIR, "retention.csv")
    if os.path.exists(retention_path):
        retention = pd.read_csv(retention_path)[["student_id", "returned_year_2"]]
        joined = student_agg.merge(retention, on="student_id")
        joined["sentiment_band"] = pd.cut(
            joined["mean_sentiment"],
            bins=[-1.01, -0.2, 0.0, 0.2, 1.01],
            labels=["Very Negative", "Slightly Negative", "Slightly Positive", "Positive"],
        )
        cross = (joined.groupby("sentiment_band", observed=True)
                       .agg(n=("student_id", "count"),
                            retention_rate=("returned_year_2", "mean"))
                       .reset_index())
        cross.to_csv(os.path.join(ART_DIR, "sentiment_vs_retention.csv"), index=False)
        print("\nSentiment vs retention:")
        print(cross.to_string(index=False))

    print(f"\nArtifacts written to {ART_DIR}/")
    print("  evaluation_sentiment.csv, evaluation_topics.csv, "
          "student_topics.csv, sentiment_vs_retention.csv")


if __name__ == "__main__":
    main()
