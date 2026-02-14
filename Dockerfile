FROM python:3.11-slim

# ---- System deps (important for matplotlib, graphviz, pygraphviz if used) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    build-essential \
    pkg-config \
    graphviz \
    graphviz-dev \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- App directory ----
WORKDIR /app

# ---- Install Python deps first (better caching) ----
COPY requirements.txt /app/requirements.txt
#RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy project code ----
COPY . /app

# ---- Streamlit config (optional but nice) ----
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

# ---- Default command: show help ----
CMD ["bash", "-lc", "echo 'Use docker compose to run: pipeline or UI'"]
