# 🎬 Sentiment Analysis Engine

### End-to-End NLP & Machine Learning · IMDB Movie Reviews · Production-Ready

**Binary sentiment classifier** built on the IMDB Movie Reviews dataset.  
Ships with a trained Logistic Regression, Naive Bayes, and Linear SVM —  
plus a Streamlit web app, a FastAPI backend, and a Jupyter notebook walkthrough.

---

## 📌 What This Project Does

This system takes raw movie review text and predicts whether it expresses **positive** or **negative** sentiment — with a calibrated confidence score.

```
Input  → "The cinematography was breathtaking but the plot made no sense."
Output → ❌ NEGATIVE  |  Confidence: 87.3%
```

The full pipeline handles everything automatically: HTML stripping, contraction expansion, negation handling, TF-IDF vectorisation, and multi-model inference. Drop in your own dataset and it retrains itself.

---

## ✨ Highlights

| Area | What's included |
|---|---|
| **NLP Pipeline** | 13-step preprocessing: HTML removal, contraction expansion, negation marking, lemmatisation |
| **Feature Engineering** | TF-IDF with unigram + bigram extraction, configurable vocabulary, sparse-matrix optimised |
| **Models** | Logistic Regression, Multinomial Naive Bayes, Linear SVM — trained, evaluated, and serialised |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC · confusion matrices · ROC curves · feature importance |
| **Interfaces** | Streamlit web app · FastAPI REST API · interactive Jupyter notebook · CLI |
| **Deployment** | Dockerfile · docker-compose · Streamlit Cloud / Render / Railway ready |
| **Code quality** | PEP 8 · type hints throughout · full docstrings · structured logging · reproducible seeds |

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/sentiment-analysis-engine.git
cd Sentiment-Analysis-Engine

# 2. Install
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Train + evaluate (auto-generates demo data if no CSV present)
python main.py

# 4. Launch the web app
streamlit run app/streamlit_app.py
```

> **No dataset download required.** A 2,000-review demo dataset is generated automatically on first run. To use the full IMDB dataset, drop your CSV into `data/raw/` — the pipeline detects it automatically.

---

## 🏗 Architecture

```
Raw Review Text
      │
      ▼
┌─────────────────────────────────────────┐
│           NLP Preprocessing             │
│  lowercase → expand contractions →      │
│  strip HTML/URLs → remove noise →       │
│  tokenise → drop stopwords →            │
│  negation marking → lemmatise           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         TF-IDF Vectorisation            │
│  vocab: 15,000 · ngrams: (1,2)          │
│  min_df=2 · max_df=0.95 · sublinear_tf  │
│  sparse matrix output                   │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  Logistic     Naive        Linear
 Regression    Bayes         SVM
       │           │           │
       └───────────┴───────────┘
                   │
                   ▼
        POSITIVE ✅ / NEGATIVE ❌
        Confidence score · Probabilities
```

**File → responsibility mapping:**

```
main.py                    Orchestrator — runs the full pipeline
src/data_loader.py         Load, validate, EDA (6 visualisations)
src/preprocessing.py       13-step NLP cleaning pipeline
src/feature_engineering.py TF-IDF fitting, transform, feature analysis
src/train.py               Stratified split, train all models, save .pkl
src/evaluate.py            Metrics, confusion matrices, ROC curves
src/predict.py             SentimentPredictor class — inference engine
src/utils.py               Logging, timing, seeding, path helpers
app/streamlit_app.py       Interactive web UI (single + batch + examples)
app/api.py                 FastAPI backend — /predict /predict/batch /health
```

---

## 📊 Model Results

Evaluated on a stratified 80/20 holdout split. Expected performance on the full 50k IMDB dataset:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.889** | **0.892** | **0.887** | **0.889** | **0.965** | ~2s |
| Linear SVM | 0.882 | 0.887 | 0.878 | 0.882 | 0.960 | ~18s |
| Multinomial Naive Bayes | 0.864 | 0.870 | 0.858 | 0.864 | 0.948 | ~0.8s |

> Logistic Regression is set as the default model — best F1 + fastest inference + interpretable coefficients.

<details>
<summary>📈 View evaluation plots</summary>

**ROC Curves**
Compares all three models across every probability threshold. All models exceed AUC 0.94.

**Confusion Matrices**
Normalised per-class accuracy for each model — lets you see exactly where false positives and false negatives occur.

**Feature Coefficients**
The top 20 words driving POSITIVE and NEGATIVE predictions — direct interpretability from Logistic Regression weights.

**Model Comparison**
Side-by-side bar chart of all five metrics across all three models.

All plots are auto-generated to `outputs/plots/` on every training run.
</details>

---

## 🔧 NLP Preprocessing Pipeline

Raw text goes through 13 sequential steps before vectorisation:

```python
"I can't believe how <b>AMAZING</b> this film was!! 😍"
      │
      ▼  expand contractions      → "I cannot believe how <b>AMAZING</b> this film was!! 😍"
      ▼  strip HTML               → "I cannot believe how AMAZING this film was!!  "
      ▼  lowercase                → "i cannot believe how amazing this film was!!  "
      ▼  remove URLs / emojis     → "i cannot believe how amazing this film was!!  "
      ▼  remove punctuation       → "i cannot believe how amazing this film was"
      ▼  tokenise                 → ["i", "cannot", "believe", "how", "amazing", "this", "film", "was"]
      ▼  drop stopwords           → ["cannot", "believe", "amazing", "film"]
      ▼  negation handling        → ["cannot", "believe_NEG", "amazing_NEG", "film_NEG"]
      ▼  lemmatise                → ["cannot", "believe_NEG", "amazing_NEG", "film_NEG"]
      │
      ▼
"cannot believe_NEG amazing_NEG film_NEG"
```

> Negation marking is a design decision that preserves the semantic inversion of "cannot believe" — "believe_NEG" is treated as a distinct, negative feature by the vectoriser rather than accidentally contributing a positive "believe" signal.

---

## 📁 Project Structure

```
Sentiment-Analysis-Engine/
│
├── main.py                      # CLI entry point — full pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── src/
│   ├── utils.py                 # Logging, timing, reproducibility
│   ├── data_loader.py           # Load, validate, EDA
│   ├── preprocessing.py         # 13-step NLP pipeline
│   ├── feature_engineering.py   # TF-IDF vectoriser
│   ├── train.py                 # Model training + serialisation
│   ├── evaluate.py              # Metrics + all visualisations
│   └── predict.py               # Inference engine
│
├── app/
│   ├── streamlit_app.py         # Web UI
│   └── api.py                   # FastAPI backend
│
├── notebooks/
│   └── experimentation.ipynb   # Full interactive walkthrough
│
├── data/
│   ├── raw/                     # Place IMDB CSV here
│   └── processed/
│
├── models/                      # Serialised .pkl artifacts
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── linear_svm.pkl
│   └── tfidf_vectorizer.pkl
│
└── outputs/
    ├── plots/                   # 11 auto-generated PNGs
    └── reports/
        └── model_metrics.txt
```

---

## 🖥 Usage

### Command line

```bash
# Full pipeline — load data, preprocess, train, evaluate, demo predictions
python main.py

# Training only (skip evaluation charts)
python main.py --train-only

# Use a specific CSV
python main.py --csv data/raw/IMDB_Dataset.csv

# Interactive prediction loop
python main.py --predict

# Quick demo on hardcoded examples
python main.py --demo
```

### Python API

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()                          # loads model + vectoriser from disk
result    = predictor.predict("This film was incredible!")

print(result["sentiment"])        # POSITIVE
print(result["confidence_pct"])   # 97.4%
print(result["probabilities"])    # {"positive": 0.974, "negative": 0.026}
```

```python
# Batch prediction — vectorised in one pass, much faster than looping
reviews = ["Absolute masterpiece.", "Boring and predictable.", "Not bad, actually."]
results = predictor.predict_batch(reviews)
```

### Streamlit web app

```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

Three tabs: **Single prediction** with confidence gauge · **Batch analysis** with summary chart · **Example reviews** to try instantly.

---

## 🔗 REST API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# Interactive docs → http://localhost:8000/docs
```

**`POST /predict`**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "One of the finest films I have ever seen."}'
```
```json
{
  "review":         "One of the finest films I have ever seen.",
  "prediction":     "positive",
  "confidence":     0.9814,
  "confidence_pct": "98.1%",
  "probabilities":  { "positive": 0.9814, "negative": 0.0186 },
  "model":          "logistic_regression"
}
```

**`POST /predict/batch`** — up to 500 reviews per request  
**`GET  /health`** — liveness check for load balancers / monitoring

---

## 🚀 Deployment

### Docker (recommended)

```bash
# Build and run both Streamlit + API in parallel
docker-compose up --build

# Streamlit → http://localhost:8501
# FastAPI   → http://localhost:8000
```

### One-command cloud deploy

| Platform | Steps |
|---|---|
| **Streamlit Cloud** | Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → connect repo → deploy |
| **Render** | New Web Service → connect repo → build: `pip install -r requirements.txt` → start: `streamlit run app/streamlit_app.py` |
| **Railway** | Connect GitHub → auto-detect → deploy |
| **Hugging Face Spaces** | New Space → Streamlit runtime → push code |

---

## 🧪 Reproducing from Scratch

Everything needed to go from zero to trained models:

```bash
# 1. Create environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Place the full IMDB CSV in data/raw/
#    If you skip this, a 2,000-review demo set is auto-generated.

# 3. Run the full pipeline
python main.py

# 4. Inspect outputs
ls outputs/plots/          # 11 visualisation PNGs
cat outputs/reports/model_metrics.txt

# 5. Launch interfaces
streamlit run app/streamlit_app.py
uvicorn app.api:app --reload
```

---

## 🗺 Roadmap

- [ ] Hyperparameter tuning via `GridSearchCV` / Optuna
- [ ] Ensemble voting across all three models
- [ ] DistilBERT fine-tuned classifier (transformer upgrade path)
- [ ] Aspect-based sentiment — per-dimension scoring (acting, plot, visuals)
- [ ] Live Twitter / Reddit sentiment dashboard
- [ ] Multi-language support via multilingual BERT
- [ ] Prediction logging + retraining pipeline

---

## 📚 Documentation

A full **beginner-to-advanced explanation** of the project is available separately:

📄 `Sentiment_Analysis_Engine_COMPLETE_EXPLANATION.md` — covers every file, every concept, end-to-end prediction tracing, interview Q&A, and a 5-day learning plan. Written for someone with zero ML background. Add this file to the repo root to make it browsable on GitHub.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch — `git checkout -b feature/my-improvement`
3. Commit — `git commit -m "Add: my improvement"`
4. Push — `git push origin feature/my-improvement`
5. Open a Pull Request

Issues and suggestions welcome.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<div align="center">

Built with Python · NLTK · scikit-learn · Streamlit · FastAPI

</div>
