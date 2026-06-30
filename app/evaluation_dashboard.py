import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import OUTPUTS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

def show_model_comparison():
    st.divider()

    st.header("📊 Model Evaluation Dashboard")

    csv_file = OUTPUTS_DIR / "model_comparison.csv"

    if not csv_file.exists():
            st.warning("Train the models first.")
            return

    df = pd.read_csv(csv_file)
    
    st.subheader("Performance Summary")

    st.dataframe(
        df,
        use_container_width=True,
    )
    
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df["Model"], df["Accuracy"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0,1)
    st.pyplot(fig)
    
    st.subheader("Precision Comparison")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df["Model"], df["Precision"])
    ax.set_ylim(0,1)
    st.pyplot(fig)
    
    st.subheader("Recall Comparison")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df["Model"], df["Recall"])
    ax.set_ylim(0,1)
    st.pyplot(fig)
    
    st.subheader("F1 Score Comparison")
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(df["Model"], df["F1"])
    ax.set_ylim(0,1)
    st.pyplot(fig)
    
    best = df.sort_values("Accuracy", ascending=False).iloc[0]

    st.success(
        f"""
        Model: {best['Model']} 
        Accuracy: {best['Accuracy']:4f}
        Precision: {best['Precision']:4f}
        Recall: {best['Recall']:4f}
        F1 : {best['F1']:4f}
        """
        )