# mlops-pytorch-pipeline

**Assignment 3 — MLOps & Infrastructure for Machine Learning**
**Roll number:** da25m539

A PyTorch image classification pipeline (CIFAR-10, ResNet-18) taken through the full deployment
lifecycle: local training, multi-stage Docker builds, and orchestrated deployment on Kubernetes
with a FastAPI serving layer.

## Status

Work in progress. See `configs/`, `src/`, `docker/`, and `k8s/` as they are filled in across PRs.

## Project layout

```
mlops-pytorch-pipeline-a3-da25m539/
├── README.md
├── .gitignore
├── .github/workflows/ci.yml   # lint + test on push/PR
├── src/
│   ├── train.py                # training loop (config-driven, JSON-line logs, early stopping)
│   ├── model.py                # ResNet-18 / CNN classifier
│   ├── dataset.py               # CIFAR-10 loading + transforms
│   └── serve.py                 # FastAPI inference service
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Setup

Instructions will be filled in as each stage lands (local training, Docker build/run, Kubernetes
deployment on Minikube).

## Architecture

Diagram to be added once the pipeline is complete end-to-end.

## Git workflow

- `main` — stable, merged work only
- `develop` — integration branch
- `feature/*` — one branch per unit of work, merged into `develop` via PR
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

## Reflection

_(300–500 word write-up added in the final PR.)_
