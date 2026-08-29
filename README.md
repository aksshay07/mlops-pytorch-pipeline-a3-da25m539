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

First you need Python 3.11, Docker Desktop, kubectl and minikube installed on your machine. I used brew to install kubectl and minikube on mac, docker desktop already comes with its own kubectl but I just used the homebrew one.

To run training locally without docker, first make a venv and install stuff:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt
```
then just run cd src and python train.py, it will read configs/training_config.yaml and start training on cifar10, it downloads the dataset automatically the first time so it will be slow initially.

For serving locally same thing but install requirements/serve.txt instead and run python serve.py, it starts a fastapi server on port 8080.

To run using docker:
```
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v1
```
and for serving
```
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints mlops-serve:v1
```
then you can curl localhost:8080/health or send a POST to /predict with an image.

To run on kubernetes (we used minikube locally), start minikube first with enough memory, we found the default was too small and training got stuck because of it:
```
minikube start --memory=6144 --cpus=4
```
then load both images into minikube since they are only built locally and not pushed to a registry anywhere:
```
minikube image load mlops-train:v1
minikube image load mlops-serve:v1
```
then apply everything in k8s folder in this order, namespace first then configmap then pvc then the training job, wait for the job to finish (takes a couple hours on cpu only since minikube has no gpu access), then apply the serving deployment, service and hpa. after that you can port forward the service and curl it same as above.

## Architecture

its basically like this, pretty simple flow

```
your laptop
   |
   |  train.py reads configs/training_config.yaml
   |  loads cifar10 using dataset.py
   |  builds resnet18 using model.py
   v
checkpoint file (classifier_v1.pt)
   |
   |  gets baked into docker image OR mounted from a volume/pvc
   v
serve.py (fastapi app)
   |
   |  loads the checkpoint at startup
   |  exposes /health and /predict
   v
kubernetes deployment (2 replicas) --> service --> hpa scales replicas based on cpu
```

so training and serving are basically two separate paths that both start from the same src code, training produces the checkpoint and serving consumes it. docker just packages each of these into an image, and kubernetes is just running multiple copies of the serving image behind a load balancer (the service) with autoscaling on top (hpa), while the training job runs once as a batch job not a long running thing.

## Git workflow

- `main` — stable, merged work only
- `develop` — integration branch
- `feature/*` — one branch per unit of work, merged into `develop` via PR
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
