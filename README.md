# Emoji-Based Mood Analyzer

Detect the emotional tone behind any text using emoji signals + NLP.

## Features
- **6-class Ekman emotion classifier** (Joy · Sadness · Anger · Fear · Surprise · Disgust)
- **Emoji signal breakdown** — explains which emojis drove the prediction
- **Mood history & trends** — timeline + distribution charts
- **Well-being alert** — flags persistent negative mood streaks
- **Model comparison** — Logistic Regression, SVM, Random Forest, Gradient Boosting

## Project Structure
```
emoji_mood_analyzer/
├── app.py                    # Streamlit app
├── model/
│   ├── train.py              # Feature engineering + training pipeline
│   ├── predict.py            # Inference + well-being check
│   └── artifacts/model.pkl   # Trained model (auto-generated)
├── data/
│   ├── generate_data.py      # Dataset generator
│   └── sample_data.csv       # 120 labeled samples (6 classes × 20)
├── notebook/
│   └── analysis.ipynb        # EDA + model training walkthrough
└── requirements.txt
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Or open the notebook
jupyter notebook notebook/analysis.ipynb
```

## Model Details
| Model               | CV F1 (macro) |
|---------------------|---------------|
| Gradient Boosting   | 0.966         |
| Random Forest       | 0.932         |
| Logistic Regression | 0.857         |
| SVM (RBF)           | 0.821         |

**Pipeline:** TF-IDF (word + bigrams on demojized text) → Gradient Boosting  
**Key feature:** Emojis are converted to Unicode names (e.g., 😊 → `smiling_face_with_smiling_eyes`) before TF-IDF, making them first-class text features.

## Tech Stack
Python · Pandas · Scikit-learn · Matplotlib · Seaborn · Streamlit
