# AI / ML Engineering Demo Repository

Collection of practical AI and machine learning demos covering:

- Retrieval-Augmented Generation (RAG)
- MLOps architecture and experiment tracking
- Time-series forecasting
- Databricks + MLflow integrations
- LangChain applications
- Dockerized development environments
- CI workflows for the main branch

---

# Repository Structure

## 1. RAG Tutorials + MLOps Architecture
Hands-on Retrieval-Augmented Generation demos and supporting MLOps workflows.

Includes:
- vector retrieval pipelines
- document chunking and embeddings
- LangChain integrations
- architecture diagrams and workflows
- containerized development setup

---

## 2. Time Series Forecasting with MLflow
Healthcare claims and call-center forecasting demo using:
- Random Forest regression
- lag and rolling-window features
- MLflow experiment tracking
- forecasting visualizations
- prediction artifact logging

---

## 3. LangChain + Databricks + MLflow RAG Demo
End-to-end RAG experimentation with:
- LangChain
- Databricks MLflow tracking
- experiment logging
- model artifact management
- classification and retrieval workflows

---

# Tech Stack

- Python
- PyTorch
- LangChain
- MLflow
- Databricks
- Docker
- GitHub Actions / CI
- Scikit-learn
- Pandas
- Jupyter Notebook

---

# Docker Setup

## 1. Build Docker Image

```bash
docker build --no-cache --progress=plain -t rag-gpu-jupyter .
```

## 2. Run Docker Container

```bash
docker run --gpus all -it -p 8008:8008 -v "$(pwd):/notebooks" rag-gpu-jupyter
```

After the container starts, a Jupyter Notebook URL will be generated in the terminal.

---

# CI / Automation

This repository includes CI workflows for:
- automated validation on main branch
- reproducible environments
- containerized execution
- experiment reproducibility

---

# Use Cases

- RAG experimentation
- MLOps demonstrations
- MLflow tracking examples
- Databricks integration demos
- Time-series forecasting
- Local GPU development with Docker
- AI engineering interview / portfolio projects