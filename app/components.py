import streamlit as st


def hero():

    st.markdown("""
    <div class="hero">

    <h1>🎬 Sentiment Analysis Engine</h1>

    <p>
    Production Ready NLP Pipeline using
    TF-IDF + Machine Learning
    </p>

    </div>
    """,
    unsafe_allow_html=True)


def stats():

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.markdown("""
        <div class="card">
        <div class="metric">50K+</div>
        <div class="label">Movie Reviews</div>
        </div>
        """,unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <div class="metric">3</div>
        <div class="label">ML Models</div>
        </div>
        """,unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <div class="metric">TF-IDF</div>
        <div class="label">Feature Engineering</div>
        </div>
        """,unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card">
        <div class="metric">Python</div>
        <div class="label">Scikit-Learn</div>
        </div>
        """,unsafe_allow_html=True)