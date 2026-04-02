import pickle
import re
import unicodedata
from collections import Counter
from model.train import compute_emoji_features, emoji_to_text, extract_emojis, is_emoji


MOOD_META = {
    "joy":      {"color": "#1D9E75", "bg": "#E1F5EE", "icon": "😊"},
    "sadness":  {"color": "#185FA5", "bg": "#E6F1FB", "icon": "😢"},
    "anger":    {"color": "#993C1D", "bg": "#FAECE7", "icon": "😡"},
    "fear":     {"color": "#854F0B", "bg": "#FAEEDA", "icon": "😨"},
    "surprise": {"color": "#534AB7", "bg": "#EEEDFE", "icon": "😲"},
    "disgust":  {"color": "#3B6D11", "bg": "#EAF3DE", "icon": "🤢"},
}

WELLBEING_MOODS = {"sadness", "fear", "anger", "disgust"}
WELLBEING_THRESHOLD = 3  # consecutive negative mood count to trigger alert


def load_model(path: str = "model/artifacts/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(text: str, artifacts: dict) -> dict:
    """
    Full prediction pipeline.
    Returns mood, confidence, probabilities per class, emoji breakdown.
    """
    pipeline = artifacts["pipeline"]
    le = artifacts["label_encoder"]
    emoji_sentiment = artifacts["emoji_sentiment"]

    cleaned = emoji_to_text(text)
    cleaned = re.sub(r"[^\w\s:]", " ", cleaned).lower().strip()

    proba = pipeline.predict_proba([cleaned])[0]
    pred_idx = proba.argmax()
    pred_mood = le.inverse_transform([pred_idx])[0]
    confidence = round(float(proba[pred_idx]) * 100, 1)

    all_probs = {
        le.inverse_transform([i])[0]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
    }

    emojis_in_text = extract_emojis(text)
    emoji_breakdown = []
    for e in emojis_in_text:
        if e in emoji_sentiment:
            mood, score = emoji_sentiment[e]
            try:
                name = unicodedata.name(e, e).lower().replace(" ", "_")
            except Exception:
                name = "emoji"
            emoji_breakdown.append({
                "emoji": e,
                "name": name,
                "mood_signal": mood,
                "weight": round(score, 2),
            })

    ef = compute_emoji_features(text)

    return {
        "text": text,
        "predicted_mood": pred_mood,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "emoji_features": ef,
        "emoji_breakdown": emoji_breakdown,
        "mood_meta": MOOD_META.get(pred_mood, {}),
    }


def check_wellbeing(history: list[str]) -> dict:
    """
    Given a list of recent predicted moods (oldest→newest),
    return a well-being alert if negative moods dominate.
    """
    if not history:
        return {"alert": False}

    recent = history[-5:]  # last 5
    negative_count = sum(1 for m in recent if m in WELLBEING_MOODS)
    streak = 0
    for m in reversed(recent):
        if m in WELLBEING_MOODS:
            streak += 1
        else:
            break

    alert = streak >= WELLBEING_THRESHOLD
    dominant = Counter(recent).most_common(1)[0][0]

    return {
        "alert": alert,
        "streak": streak,
        "negative_ratio": round(negative_count / len(recent), 2),
        "dominant_recent_mood": dominant,
        "message": (
            "Persistent negative mood pattern detected. Consider taking a break."
            if alert else "Mood pattern looks okay."
        ),
    }
