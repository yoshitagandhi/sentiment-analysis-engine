# Multi-stage Dockerfile for Sentiment Analysis Engine
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Download all required NLTK resources (punkt_tab required for NLTK 3.9+)
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

# Expose ports
EXPOSE 8501 8000

# Default: Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
