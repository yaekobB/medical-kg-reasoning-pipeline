# Medical Knowledge‑Graph Reasoning (Docker + Streamlit UI)

A compact end‑to‑end pipeline that:
1) preprocesses a medical dataset (CSV)  
2) builds an **instance RDF knowledge graph (KG)**  
3) runs **SHACL validation**  
4) runs **RDFS reasoning** and compares RAW vs REASONED graphs  
5) runs **OWL‑RL reasoning** for an **RDFS limitation demo**  
6) provides a **Streamlit UI** for exploration + query demos + LLM‑assisted explanations (Ollama)

---

## Features

- **Data → RDF KG** generation (Turtle `.ttl`)
- **SHACL validation** (RAW and inferred KG)
- **RDFS closure reasoning**
- **RAW vs REASONED comparison** (triple deltas, type deltas, queries that change)
- **OWL‑RL reasoning** (to demonstrate what RDFS cannot infer)
- **Streamlit UI** (runs in Docker, exposed on port `8501`)
- Optional: import inferred KG to **Neo4j** (script provided)

---

## Project structure

```
.
├─ docker-compose.yml
├─ Dockerfile
├─ docker/entrypoint.sh
├─ run_all.sh
├─ requirements.txt
├─ README.md
├─ import_medical_kg_to_neo4j.py
├─ data/
│  ├─ raw/         # original dataset
│  ├─ processed/   # generated CSV + TTL files + reports
│  └─ ontology/    # schema + SHACL shapes + OWL-RL extensions
└─ src/
   ├─ run_pipeline.py
   ├─ preprocessing/
   ├─ rdf_build/
   ├─ validation/
   ├─ reasoning/
   ├─ sparql/
   ├─ visualization/
   ├─ analysis/
   ├─ ui_prototypes/app_medium.py
   ├─ llm_simple/
   └─ llm_large/
```

---

## Outputs you should expect

After a successful run, you will see (paths are inside the container, but persisted to your local `./data` folder via volume):

- `data/processed/diseases_large.csv`
- `data/processed/data_medical_large.ttl` (RAW instance KG)
- `data/processed/data_medical_large_inferred.ttl` (RDFS inferred KG)
- `data/processed/data_medical_large_owlrl_inferred.ttl` (OWL‑RL inferred KG)
- `data/processed/rdfs_limitations_report.txt`
- `data/visualizations/*.png`

---

## Prerequisites

### Option A - Run everything with Docker (recommended)
- Docker Desktop installed and running

### Option B - Run locally (without Docker)
- Python 3.10+ (3.11 also fine)
- (Optional for LLM tabs) **Ollama** installed and a model pulled (e.g., `llama3.2`)

---

## Quickstart (Docker)

### 1) Start the pipeline + Streamlit UI

From the project root:

```bash
docker compose up --build
```

Then open your browser:

- **http://localhost:8501**

You should see the Streamlit UI and tabs for queries, reasoning comparisons, and LLM features.

### 2) Stop everything

```bash
docker compose down
```

---

## Docker configuration (recommended compose)

This is the typical setup to run **pipeline + UI**:

```yaml
services:
  medicalkg:
    build: .
    container_name: medicalkg
    working_dir: /app

    volumes:
      - ./data:/app/data
      - venv:/app/.venv

    ports:
      - "8501:8501"

    environment:
      RUN_SHACL: "1"
      RUN_SHACL_INFERRED: "1"
      RUN_RDFS_LIMIT: "1"
      RUN_STREAMLIT: "1"

    entrypoint: ["/app/docker/entrypoint.sh"]
    command: ["ui"]

volumes:
  venv:
```

> Notes:
> - `./data:/app/data` persists all generated artifacts to your local machine.
> - `venv:/app/.venv` caches dependencies inside a docker volume for faster restarts.

---

## How the Docker entrypoint works

The container entrypoint supports two modes:

- `pipeline` → run end‑to‑end pipeline only
- `ui` → run pipeline and launch Streamlit UI

Examples:

```bash
# pipeline only
docker compose run --rm medicalkg pipeline

# pipeline + UI
docker compose run --rm -p 8501:8501 medicalkg ui
```

With the provided `docker-compose.yml`, you usually only need:

```bash
docker compose up --build
```

---

## Local run (without Docker)

### 1) Run the pipeline

```bash
bash run_all.sh
```

### 2) Run pipeline + UI

```bash
bash run_all.sh --with-ui
```

Then open:

- **http://localhost:8501**

---

## LLM tabs (Ollama) - what you need to know

Your UI contains two LLM approaches:

- **LLM Simple tab**: lightweight prompt + KG context → Ollama call
- **LLM Large tab**: a more structured/robust integration (intent parsing + KG API + explanation pipeline)

### Running Ollama locally (host machine)
Install Ollama and pull a model once:

```bash
ollama pull llama3.2
ollama serve
```

### Important: Docker container → accessing Ollama on the host
If the Streamlit UI runs **inside Docker** but Ollama runs on your **host**, your LLM client must reach the host service.

Common working base URL patterns:

- **Windows/macOS Docker Desktop:** `http://host.docker.internal:11434`
- **Linux:** you may need `http://172.17.0.1:11434` or use host networking (varies)

If your code uses an `OLLAMA_BASE_URL`, set it to the correct host address for Docker.
(Your latest update already made both LLM modes work in Docker — keep the same setting.)

---

## Troubleshooting

### UI starts but you cannot open it
- Confirm port mapping exists: `8501:8501`
- Open: `http://localhost:8501` (not `0.0.0.0:8501`)

### Streamlit says “Address already in use”
Stop previous containers or processes:
```bash
docker compose down
```

### LLM tabs fail (connection error)
- Ensure Ollama is running on the host: `ollama serve`
- Ensure the container can reach it:
  - Use `host.docker.internal` on Windows/macOS Docker Desktop
  - Confirm your `OLLAMA_BASE_URL` inside the container points to the host

### Pipeline runs but outputs are missing on your machine
- Check the volume mapping:
  - `./data:/app/data`
- All generated files are written under `data/processed/` and `data/visualizations/`.

---

## License / academic note

This project is intended for academic and research demonstration.  
**It is NOT a medical diagnosis or treatment tool.** Always consult qualified professionals for real medical decisions.
