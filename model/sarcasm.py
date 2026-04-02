"""
Lightweight sarcasm / irony detector.

Approach: Rule-based signals + TF-IDF + Logistic Regression.
Trained on synthetic patterns since we have no labeled sarcasm dataset,
but the feature engineering captures the real linguistic signals.

Key signals:
  - Positive words + negative emojis (or vice versa)
  - Exaggerated punctuation  (!! ??? ...)
  - ALL CAPS words
  - "oh great", "yeah right", "sure sure", "totally", "obviously"
  - Emoji-text sentiment mismatch
"""

import re
import pickle
import os
import random
import unicodedata

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

random.seed(99)

# ─── Sarcasm lexicon signals ────────────────────────────────────────────────
SARCASM_PHRASES = [
    "oh great", "yeah right", "sure sure", "totally fine",
    "obviously", "wow thanks", "oh wow", "love that for me",
    "how wonderful", "just perfect", "so helpful", "amazing right",
    "not like", "what a surprise", "who would have thought",
    "oh because that makes sense", "brilliant idea", "genius",
    "so funny", "as if", "oh because", "sure jan",
]

POSITIVE_WORDS = {
    "great","amazing","wonderful","perfect","love","best","fantastic",
    "awesome","brilliant","excellent","superb","nice","good","fine",
    "happy","glad","pleased","thrilled","excited","beautiful"
}

NEGATIVE_EMOJIS_RANGE = [
    (0x1F620, 0x1F625), (0x1F628, 0x1F62D),
    (0x1F610, 0x1F611), (0x1F644, 0x1F644),
]

POSITIVE_EMOJIS = {"😊","😄","😁","😃","🥰","😍","🤩","😂","🤣","🥳","🎉","🎊","✨","💛","🙌"}
NEGATIVE_EMOJIS = {"😒","🙄","😑","😐","😤","😠","😡","🤬","💢","😞","😔","😢","😭","💔"}


def _is_negative_emoji(ch: str) -> bool:
    cp = ord(ch)
    return any(s <= cp <= e for s, e in NEGATIVE_EMOJIS_RANGE) or ch in NEGATIVE_EMOJIS


def extract_sarcasm_features(text: str) -> dict:
    lower = text.lower()
    words = lower.split()
    upper_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    pos_words_found = [w for w in words if re.sub(r"[^a-z]", "", w) in POSITIVE_WORDS]

    neg_emoji_count = sum(1 for ch in text if _is_negative_emoji(ch))
    pos_emoji_count = sum(1 for ch in text if ch in POSITIVE_EMOJIS)

    # Core mismatch: positive words + negative emojis
    sentiment_mismatch = int(
        (len(pos_words_found) > 0 and neg_emoji_count > 0) or
        (pos_emoji_count > 0 and any(w in lower for w in ["hate","awful","terrible","worst","bad","horrible"]))
    )

    phrase_hits = sum(1 for p in SARCASM_PHRASES if p in lower)
    caps_ratio = len(upper_words) / max(len(text.split()), 1)
    excl_count = text.count("!!")
    quest_count = text.count("??")
    ellipsis_count = text.count("...")

    return {
        "sentiment_mismatch": sentiment_mismatch,
        "sarcasm_phrase_hits": phrase_hits,
        "caps_ratio": round(caps_ratio, 3),
        "double_excl": excl_count,
        "double_quest": quest_count,
        "ellipsis": ellipsis_count,
        "neg_emoji_with_pos_text": int(len(pos_words_found) > 0 and neg_emoji_count > 0),
    }


def _sarcasm_score(feats: dict) -> float:
    score = 0.0
    score += feats["sentiment_mismatch"] * 0.45
    score += feats["sarcasm_phrase_hits"] * 0.25
    score += feats["caps_ratio"] * 0.15
    score += min(feats["double_excl"], 3) * 0.05
    score += min(feats["double_quest"], 3) * 0.05
    score += feats["neg_emoji_with_pos_text"] * 0.35
    return round(min(score, 1.0), 3)


# ─── Synthetic training data ─────────────────────────────────────────────────
def _build_sarcasm_training_data():
    sarcastic = [
        "Oh great, another Monday 😒",
        "Yeah RIGHT, like that's ever going to happen 🙄",
        "Totally fine, everything is FINE 😤",
        "Oh wow, what a BRILLIANT idea 😑",
        "Sure sure, I totally believe you 😒💔",
        "How WONDERFUL, my code broke again 😠",
        "Love that for me, truly 🙄",
        "Oh because that makes SO much sense 😤",
        "Amazing, just perfect timing as always 😒",
        "WOW thanks SO much for the help 😡",
        "Oh GREAT the wifi is down again 😑",
        "Genius move, truly a genius 🙄😒",
        "Yeah because I totally needed more work 😤",
        "Oh sure, blame me again, why not 😠",
        "Not like I had ANYTHING else to do 😒",
        "How nice, cancelled at the last minute AGAIN 😑",
        "Obviously, who would have thought 🙄",
        "Sure Jan, totally believable 😒",
        "What a surprise, the bus is late AGAIN 😤",
        "Oh wow I am SO thrilled about this 😑💢",
        "Fantastic, my favourite coffee mug broke 😒",
        "As if that was ever going to work out 🙄",
        "Oh I'm SO glad you told me at the last second 😠",
        "Brilliant, just brilliant planning there 😤😒",
        "Yeah right, I'm sure that will go perfectly 😑",
        "Oh wonderful, more meetings added to my calendar 😒",
        "Super helpful advice, truly life changing 🙄",
        "How lovely, rained on the one day I forgot my umbrella 😤",
        "Oh great, printer jammed AGAIN right before the deadline 😠",
        "So funny how nothing ever goes right 😒💔",
    ]

    sincere = [
        "I am so happy today 😊",
        "This is genuinely the best day ever 🎉",
        "Really grateful for all your help 😊💛",
        "Feeling great after that workout 💪😄",
        "Amazing news, I got the job!! 🥳",
        "Truly wonderful to see everyone 😍",
        "Had such a fantastic time last night 😃✨",
        "So excited for this trip 🎊😄",
        "Genuinely love this new place 😊",
        "Best birthday ever, thank you all 🥰🎂",
        "I actually really enjoyed that movie 😁",
        "Feeling so refreshed after the holiday 😊",
        "Honestly the most fun I have had in ages 😂🤣",
        "Really proud of what the team achieved 🙌",
        "This food is incredible, highly recommend 😍",
        "So thankful for my friends 💛😊",
        "Woke up in the best mood today 😄",
        "That sunset was genuinely breathtaking 🌅😊",
        "Passed my exam, so relieved and happy 😁✨",
        "Finally finished the project, feels amazing 🙌😃",
        "Really love spending time with family 🥰",
        "That performance was genuinely outstanding 😮✨",
        "Feeling motivated and ready to go 💪😊",
        "So pleased with how it turned out 😄",
        "Had the most wonderful evening 😊🌙",
        "Really appreciating the small things today 💛",
        "This song makes me so happy every time 😁🎵",
        "Feeling genuinely content right now 😊",
        "Had a surprisingly great day 😃",
        "So proud of how far I have come 🙌✨",
    ]

    data = (
        [(t, 1) for t in sarcastic] +
        [(t, 0) for t in sincere]
    )
    random.shuffle(data)
    return data


# ─── Train & save ────────────────────────────────────────────────────────────
def train_sarcasm_model(save_dir: str = "model/artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/sarcasm_model.pkl"

    data = _build_sarcasm_training_data()
    texts = [d[0] for d in data]
    labels = [d[1] for d in data]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=2000, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=2.0, max_iter=500, random_state=42)),
    ])
    cv = cross_val_score(pipe, texts, labels, cv=5, scoring="f1")
    pipe.fit(texts, labels)

    print(f"  Sarcasm model CV F1: {cv.mean():.3f} ± {cv.std():.3f}")

    with open(save_path, "wb") as f:
        pickle.dump({"pipeline": pipe}, f)
    print(f"  Saved: {save_path}")
    return pipe


def load_sarcasm_model(save_dir: str = "model/artifacts"):
    path = f"{save_dir}/sarcasm_model.pkl"
    if not os.path.exists(path):
        print("  Training sarcasm model...")
        return train_sarcasm_model(save_dir)
    with open(path, "rb") as f:
        return pickle.load(f)["pipeline"]


def detect_sarcasm(text: str, model) -> dict:
    """
    Returns:
      is_sarcastic  bool
      confidence    float (0-100)
      score         rule-based score (0-1)
      signals       list of detected signals
    """
    feats = extract_sarcasm_features(text)
    rule_score = _sarcasm_score(feats)

    ml_proba = model.predict_proba([text])[0]
    ml_score = float(ml_proba[1])

    # Blend: 60% ML + 40% rule-based
    blended = round(0.6 * ml_score + 0.4 * rule_score, 3)
    is_sarcastic = blended >= 0.45

    signals = []
    if feats["sentiment_mismatch"]:
        signals.append("positive words + negative emojis")
    if feats["sarcasm_phrase_hits"] > 0:
        signals.append(f"sarcasm phrase detected")
    if feats["caps_ratio"] > 0.3:
        signals.append("excessive CAPS")
    if feats["double_excl"] > 0:
        signals.append("double exclamation !!")
    if feats["neg_emoji_with_pos_text"]:
        signals.append("emoji-text sentiment mismatch")

    return {
        "is_sarcastic": is_sarcastic,
        "confidence": round(blended * 100, 1),
        "rule_score": rule_score,
        "ml_score": round(ml_score * 100, 1),
        "signals": signals,
    }


if __name__ == "__main__":
    pipe = train_sarcasm_model()
    tests = [
        ("Oh great, another Monday 😒", True),
        ("Yeah right, totally believe you 🙄", True),
        ("I am so happy today 😊", False),
        ("Amazing news, I got the job!! 🥳", False),
        ("How WONDERFUL, my code broke again 😤", True),
        ("Genuinely grateful for your help 💛😊", False),
    ]
    print("\n── Sarcasm detection test ────────────────────")
    for text, expected in tests:
        result = detect_sarcasm(text, pipe)
        status = "✓" if result["is_sarcastic"] == expected else "✗"
        print(f"  {status}  [{result['confidence']:5.1f}%]  {text[:55]}")
