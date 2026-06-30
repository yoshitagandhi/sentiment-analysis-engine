from evaluation_dashboard import show_model_comparison
import sys
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st

from src.predictor import SentimentPredictor

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# -------------------------------------------------
# THEME COLORS
# -------------------------------------------------

PRIMARY = "#4F46E5"
SECONDARY = "#06B6D4"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
CARD = "#1F2937"

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown(
    f"""
<style>

/* Main App */
.main {{
    background-color: #0E1117;
}}

/* Reduce default padding */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background: #1E293B;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.20);
}}

div[data-testid="stMetric"]:hover {{
    border: 1px solid {PRIMARY};
    transition: 0.3s ease;
}}

/* Buttons */
.stButton > button {{
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg,{PRIMARY},{SECONDARY});
    color: white;
    font-size: 16px;
    font-weight: 600;
}}

.stButton > button:hover {{
    opacity: 0.92;
    transform: translateY(-2px);
    transition: 0.2s;
}}

/* Text Area */
textarea {{
    border-radius: 12px !important;
}}

/* Hero Banner */
.hero {{
    background: linear-gradient(135deg,{PRIMARY},{SECONDARY});
    border-radius: 22px;
    padding: 50px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}}

.hero h1 {{
    font-size: 48px;
    margin-bottom: 10px;
}}

.hero h3 {{
    margin-top: 0;
    font-weight: 500;
}}

.hero p {{
    font-size: 18px;
    opacity: 0.95;
}}

.footer {{
    text-align: center;
    color: gray;
    padding-top: 40px;
}}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">

<h1>🎬 Sentiment Analysis Engine</h1>

<h3>Production-Ready NLP Movie Review Classifier</h3>

<p>
Analyze movie reviews using
<b>Natural Language Processing</b>,
<b>TF-IDF Vectorization</b>,
and <b>Machine Learning</b>.
</p>

</div>
""",
    unsafe_allow_html=True,
)

badge1, badge2, badge3, badge4, badge5 = st.columns(5)

with badge1:
    st.info("🐍 Python")

with badge2:
    st.info("🧠 Scikit-Learn")

with badge3:
    st.info("📊 Streamlit")

with badge4:
    st.info("📄 TF-IDF")

with badge5:
    st.info("🎥 IMDb")
    
st.write("")  

st.markdown("## 📊 Project Overview")

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric(
        label="📄 Dataset",
        value="50,000+",
        delta="IMDb Reviews",
    )

with card2:
    st.metric(
        label="🤖 Models",
        value="3",
        delta="ML Algorithms",
    )

with card3:
    st.metric(
        label="🧠 Feature Engineering",
        value="TF-IDF",
        delta="Text Vectorization",
    )

with card4:
    st.metric(
        label="🚀 Status",
        value="Production",
        delta="Ready",
    )

st.write("")

left_info, right_info = st.columns([2, 1])

with left_info:

    st.success(
        """
### 🚀 Features

✔ End-to-End NLP Pipeline

✔ TF-IDF Feature Engineering

✔ Logistic Regression

✔ Multinomial Naive Bayes

✔ Linear Support Vector Machine

✔ Confidence Score

✔ Interactive Streamlit Dashboard
"""
    )

with right_info:

    st.info(
        """
### 📚 Project Details

**Task**

Movie Review Sentiment Analysis

**Classes**

Positive

Negative

**Dataset**

IMDb Movie Reviews

**Deployment**

Streamlit
"""
    )

st.divider()

st.markdown("# 🤖 AI Prediction Workspace")

st.caption(
    "Enter a movie review and compare predictions across multiple Machine Learning models."
)

st.write("")

left, right = st.columns([1.6, 1], gap="large")

with left:

    st.markdown("## ✍️ Input Review")

    MODEL_OPTIONS = {
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Multinomial Naive Bayes",
        "linear_svm": "Linear Support Vector Machine",
    }

    model = st.selectbox(
        "Choose Model",
        list(MODEL_OPTIONS.keys()),
        format_func=lambda x: MODEL_OPTIONS[x],
    )

    review = st.text_area(
        "Movie Review",
        key="review_text",
        height=260,
        placeholder="Paste or type a movie review here...",
    )

    predict = False
    result = None

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🚀 Analyze Review"):
            predict = True
            try:
                predictor = SentimentPredictor(model)
                result = predictor.predict(review)

            except FileNotFoundError:
                st.error(
                    "Selected model is missing. Please retrain the models using `python main.py`."
                )
                st.stop()
                
            except Exception as e:
                st.error(f"Error:{e}")
                st.stop()

    def clear_review():
        st.session_state.review_text = ""

    with col2:

        st.button(
            "🗑 Clear",
            on_click=clear_review,
            use_container_width=True,
   )
        
    st.write("")

    st.markdown("### 🎬 Quick Examples")

    ex1, ex2 = st.columns(2)

    positive_review = """
    This movie was absolutely fantastic.
    The acting was brilliant,
    the screenplay was engaging,
    and every scene kept me interested.
    Highly recommended!
    """

    negative_review = """
    This movie was terrible.
    Poor acting,
    predictable story,
    and I almost fell asleep.
    Definitely not recommended.
    """

    with ex1:

        if st.button(
            "😊 Load Positive Review",
            use_container_width=True,
       ):
            st.session_state.review_text = positive_review
            st.rerun()

    with ex2:

        if st.button(
            "☹ Load Negative Review",
            use_container_width=True,
        ):
            st.session_state.review_text = negative_review
            st.rerun()

with right:

    if not predict:
        st.info(
            "👈 Enter a movie review and click **Analyze Review**."
        )
    elif review.strip() == "":
        st.warning("Please enter a movie review to analyze.")
    else:
        with st.spinner("Analyzing review..."):
            predictor = SentimentPredictor(model)
            try:
                result = predictor.predict(review)
            except Exception as error:
                st.error(
                    "Something went wrong while analyzing the review. "
                    "Please try again or verify the review text."
                )
                st.error(str(error))
                result = None

        if result is not None:
            with right:
                st.markdown("---")

                if result["prediction"].lower() == "positive":
                    card_color = "#14532d"
                    border = "#22c55e"
                    emoji = "😊"
                    title = "POSITIVE"
                else:
                    card_color = "#7f1d1d"
                    border = "#ef4444"
                    emoji = "☹️"
                    title = "NEGATIVE"

                st.markdown(
                    f"""
<div style="
background:{card_color};
padding:28px;
border-radius:18px;
border-left:8px solid {border};
text-align:center;
color:white;
">

<h1>{emoji}</h1>

<h2>{title}</h2>

<h1>{result["confidence"]:.2f}%</h1>

<p>Prediction Confidence</p>

</div>
""",
                    unsafe_allow_html=True,
                )

                st.write("")
                st.markdown("### 📈 Prediction Confidence")
                progress_value = result["probabilities"]["Positive"]
                if progress_value > 1:
                    progress_value /= 100
                st.progress(max(0.0, min(1.0, progress_value)))

                p1, p2 = st.columns(2)

                with p1:
                    st.metric(
                        "😊 Positive",
                        f'{result["probabilities"]["Positive"]:.2f}%'
                    )

                with p2:
                    st.metric(
                        "☹ Negative",
                        f'{result["probabilities"]["Negative"]:.2f}%'
                    )

                st.write("")
                st.info(
                    f"""
### 🤖 Model Used

**{result["model"]}**

This prediction was generated using
a trained TF-IDF vectorizer
combined with the selected
Machine Learning classifier.
"""
                )

                if result["prediction"].lower() == "positive":
                    st.success(
                        """
### 💡 Interpretation

The review contains predominantly
positive language and sentiment.

The classifier predicts that
the reviewer enjoyed the movie.
"""
                    )
                else:
                    st.error(
                        """
### 💡 Interpretation

The review contains predominantly
negative language and sentiment.

The classifier predicts that
the reviewer disliked the movie.
"""
                    )

st.divider()

st.header("📂 Batch Prediction")

st.caption(
    "Upload a CSV file containing movie reviews and generate predictions."
)

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"],
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("### Preview")

    st.dataframe(
        df.head(),
        use_container_width=True,
    )

    if "review" not in df.columns:

        st.error(
            "CSV must contain a column named 'review'."
        )

    else:

        st.success(
            f"{len(df)} reviews detected."
        )             
        
        if st.button(
            "🚀 Predict Entire Dataset",
            use_container_width=True,
        ):

            predictor = SentimentPredictor(model)

            predictions = []
            confidence = []

            progress = st.progress(0)

            for i, review_text in enumerate(df["review"]):

                batch_result = predictor.predict(str(review_text))

                predictions.append(
                    batch_result["prediction"]
                )

                confidence.append(
                    batch_result["confidence"]
                )

                progress.progress(
                    (i + 1) / len(df)
                )

            df["Prediction"] = predictions

            df["Confidence"] = confidence
            
            st.success(
                "Prediction Complete!"
            )

            st.dataframe(
                df.head(),
                use_container_width=True,
            )
            
            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Predictions",
                csv,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
            st.divider()
            show_model_comparison()
    
            st.markdown("### 🤖 Model Details")

            MODEL_INFO = {
                "logistic_regression": {
                    "name": "Logistic Regression",
                    "description": "A robust linear classifier for binary sentiment prediction using TF-IDF features.",
                    "strengths": "Good generalization, easy to interpret, fast inference.",
                    "algorithm": "Linear model",
               },
            "naive_bayes": {
                "name": "Multinomial Naive Bayes",
                "description": "A probabilistic text classifier that works well with discrete word counts and TF-IDF values.",
                "strengths": "Fast training, performs well on text data, useful for noisy inputs.",
                "algorithm": "Probabilistic classifier",
            },
            "linear_svm": {
                "name": "Linear Support Vector Machine",
                "description": "A margin-based classifier that excels at separating positive and negative reviews in high-dimensional text space.",
                "strengths": "Strong performance on sparse text features, robust to overfitting.",
                "algorithm": "Support Vector Machine",
            },
    }

            details = MODEL_INFO.get(model, {})

            detail_col, summary_col = st.columns([2, 1])

            with detail_col:
                st.markdown(f"**Model:** {details.get('name', model)}")
                st.write(details.get("description", "Model information is not available."))
                st.write(f"**Best For:** {details.get('strengths', '')}")

            with summary_col:
                st.markdown("### Quick Facts")
                st.write("- **Dataset**: IMDb Movie Reviews")
                st.write("- **Classes**: Positive, Negative")
                st.write("- **Features**: TF-IDF Vectorization")
                st.write(f"- **Selected Model**: {details.get('name', model)}")
                st.write(f"- **Algorithm**: {details.get('algorithm', 'N/A')}")
                
                show_model_comparison()