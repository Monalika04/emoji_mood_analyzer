import pandas as pd
import numpy as np
import re
import pickle
import os
from collections import Counter

import unicodedata
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# ─── Emoji sentiment lexicon ────────────────────────────────────────────────
EMOJI_SENTIMENT = {
    "😊": ("joy", 0.9),       "😄": ("joy", 0.9),       "😁": ("joy", 0.85),
    "😃": ("joy", 0.85),      "🥰": ("joy", 0.95),       "😍": ("joy", 0.9),
    "🤩": ("joy", 0.95),      "😂": ("joy", 0.8),        "🤣": ("joy", 0.8),
    "🥳": ("joy", 0.9),       "🎉": ("joy", 0.85),       "🎊": ("joy", 0.85),
    "✨": ("joy", 0.75),       "💛": ("joy", 0.8),        "💪": ("joy", 0.7),
    "🙌": ("joy", 0.85),      "😢": ("sadness", 0.9),    "😭": ("sadness", 0.95),
    "😞": ("sadness", 0.85),  "😔": ("sadness", 0.8),    "💔": ("sadness", 0.9),
    "😟": ("sadness", 0.8),   "🥺": ("sadness", 0.75),   "😓": ("sadness", 0.7),
    "😡": ("anger", 0.95),    "🤬": ("anger", 0.98),     "😠": ("anger", 0.9),
    "😤": ("anger", 0.85),    "💢": ("anger", 0.8),       "👊": ("anger", 0.75),
    "✊": ("anger", 0.7),      "😰": ("fear", 0.9),       "😨": ("fear", 0.9),
    "😱": ("fear", 0.95),     "😳": ("fear", 0.7),        "🫨": ("fear", 0.85),
    "😲": ("surprise", 0.9),  "😮": ("surprise", 0.85),  "🤯": ("surprise", 0.95),
    "🤢": ("disgust", 0.9),   "🤮": ("disgust", 0.95),   "😒": ("disgust", 0.8),
    "😫": ("disgust", 0.75),  "😖": ("disgust", 0.75),
}


# ─── Feature engineering ────────────────────────────────────────────────────
EMOJI_UNICODE_RANGES = [
    (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F), (0x1F780, 0x1F7FF), (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF), (0x1FA00, 0x1FA6F), (0x1FA70, 0x1FAFF),
    (0x2600,  0x26FF),  (0x2700,  0x27BF),  (0xFE00,  0xFE0F),
    (0x1F1E0, 0x1F1FF),
]


def is_emoji(char: str) -> bool:
    cp = ord(char)
    return any(start <= cp <= end for start, end in EMOJI_UNICODE_RANGES)


def extract_emojis(text: str) -> list[str]:
    return [ch for ch in text if is_emoji(ch)]


def emoji_to_text(text: str) -> str:
    """Replace emojis with their Unicode name (portable, no library needed)."""
    result = []
    for ch in text:
        if is_emoji(ch):
            try:
                name = unicodedata.name(ch, "").lower().replace(" ", "_")
                result.append(f" {name} ")
            except Exception:
                result.append(" emoji ")
        else:
            result.append(ch)
    return "".join(result)


def compute_emoji_features(text: str) -> dict:
    emojis_found = extract_emojis(text)
    total = len(emojis_found)

    mood_votes = Counter()
    sentiment_score = 0.0

    for e in emojis_found:
        if e in EMOJI_SENTIMENT:
            mood, score = EMOJI_SENTIMENT[e]
            mood_votes[mood] += score
            sentiment_score += score

    dominant_mood = mood_votes.most_common(1)[0][0] if mood_votes else "neutral"
    dominant_score = mood_votes.most_common(1)[0][1] if mood_votes else 0.0

    return {
        "emoji_count": total,
        "unique_emojis": len(set(emojis_found)),
        "sentiment_score": round(sentiment_score, 4),
        "dominant_emoji_mood": dominant_mood,
        "dominant_score": round(dominant_score, 4),
        "emoji_density": round(total / max(len(text.split()), 1), 4),
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw text+emoji into rich feature set."""
    rows = []
    for _, row in df.iterrows():
        text = row["text"]
        ef = compute_emoji_features(text)

        # Clean text for TF-IDF (keep demojized form)
        cleaned = emoji_to_text(text)
        cleaned = re.sub(r"[^\w\s:]", " ", cleaned).lower().strip()

        rows.append({
            "text_id": row.get("text_id", 0),
            "original_text": text,
            "cleaned_text": cleaned,
            "mood": row["mood"],
            **ef,
        })
    return pd.DataFrame(rows)


# ─── Training ───────────────────────────────────────────────────────────────
def train(data_path: str = "data/sample_data.csv", save_dir: str = "model/artifacts"):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples | Classes: {df['mood'].nunique()}")

    features_df = build_feature_matrix(df)

    le = LabelEncoder()
    y = le.fit_transform(features_df["mood"])
    X_text = features_df["cleaned_text"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF pipeline (character + word n-grams to capture emoji descriptions)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
    )

    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf", C=1.5, gamma="scale", probability=True, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print("\n── Model Comparison ──────────────────────────")
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
        cv_scores = cross_val_score(pipe, X_text, y, cv=5, scoring="f1_macro")
        pipe.fit(X_train_text, y_train)
        y_pred = pipe.predict(X_test_text)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        results[name] = {"pipeline": pipe, "cv_f1": cv_scores.mean(), "test_f1": test_f1}
        print(f"  {name:<25} CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  |  Test F1: {test_f1:.3f}")

    # Pick best model by CV F1
    best_name = max(results, key=lambda k: results[k]["cv_f1"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\n  Best model: {best_name}")

    # Detailed report for best model
    y_pred_best = best_pipeline.predict(X_test_text)
    print("\n── Classification Report ─────────────────────")
    print(classification_report(
        y_test, y_pred_best,
        target_names=le.classes_,
        digits=3
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)

    # Save artifacts
    artifacts = {
        "pipeline": best_pipeline,
        "label_encoder": le,
        "model_name": best_name,
        "classes": list(le.classes_),
        "emoji_sentiment": EMOJI_SENTIMENT,
        "results_summary": {k: {"cv_f1": v["cv_f1"], "test_f1": v["test_f1"]} for k, v in results.items()},
        "confusion_matrix": cm,
    }

    with open(f"{save_dir}/model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n  Artifacts saved to {save_dir}/model.pkl")
    return artifacts


if __name__ == "__main__":
    train()
