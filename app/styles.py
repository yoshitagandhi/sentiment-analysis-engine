import streamlit as st

PRIMARY = "#6366F1"
SECONDARY = "#06B6D4"
SUCCESS = "#10B981"
DANGER = "#EF4444"
CARD = "#1E293B"
BG = "#0F172A"


def load_css():
    st.markdown("""
    <style>

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
        max-width:1200px;
    }

    .hero{
        background:linear-gradient(135deg,#4F46E5,#06B6D4);
        padding:40px;
        border-radius:20px;
        color:white;
        text-align:center;
        margin-bottom:30px;
    }

    .hero h1{
        font-size:48px;
        margin-bottom:10px;
    }

    .hero p{
        font-size:18px;
        opacity:.95;
    }

    .card{
        background:#1E293B;
        border-radius:18px;
        padding:20px;
        text-align:center;
        border:1px solid rgba(255,255,255,.08);
        transition:.2s;
    }

    .card:hover{
        transform:translateY(-5px);
    }

    .metric{
        font-size:34px;
        font-weight:700;
        color:#60A5FA;
    }

    .label{
        color:#CBD5E1;
        font-size:15px;
    }

    .prediction{
        border-radius:20px;
        padding:25px;
        margin-top:20px;
        text-align:center;
    }

    .positive{
        background:#052E16;
        border:1px solid #16A34A;
    }

    .negative{
        background:#450A0A;
        border:1px solid #DC2626;
    }

    </style>
    """, unsafe_allow_html=True)