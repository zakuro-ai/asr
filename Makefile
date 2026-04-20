.DEFAULT_GOAL := help

PACKAGE := asr_deepspeech
VERSION  := $(shell python3 -c 'from $(PACKAGE) import __version__; print(__version__)' 2>/dev/null || echo "unknown")

# Docker image names (override via env or .env file)
conf_file ?= .env
-include $(conf_file)
IMAGE_VANILLA ?= zakuroai/asr-vanilla
IMAGE_SANDBOX ?= zakuroai/asr-sandbox

DOCKER_OPTS ?= \
	--gpus all \
	--shm-size 8g \
	-v $(HOME)/.ssh:/home/user/.ssh \
	-v $(PWD):/home/user/asr \
	-v $(HOME)/data:/home/user/data

.PHONY: help install test lint typecheck build devcontainer sandbox

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install        Install package with dev extras (uv)"
	@echo "  test           Run CPU unit tests"
	@echo "  lint           Ruff lint + format check"
	@echo "  typecheck      mypy"
	@echo "  build          Build Docker images"
	@echo "  devcontainer   Open VS Code dev container"
	@echo "  sandbox        Run sandbox Docker image with JupyterLab"
	@echo ""

install:
	uv pip install -e ".[dev]"

test:
	pytest -m "not gpu and not slow" --tb=short

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy $(PACKAGE)

# Docker
build: build_docker_vanilla build_docker_sandbox

build_docker_vanilla:
	docker build . -t $(IMAGE_VANILLA) --network host -f docker/vanilla/Dockerfile

build_docker_sandbox:
	docker build . -t $(IMAGE_SANDBOX) --network host -f docker/sandbox/Dockerfile

devcontainer:
	code --folder-uri vscode-remote://dev-container+$(PWD) .

sandbox:
	@docker stop dev_$(PACKAGE)_sandbox 2>/dev/null || true
	@docker rm   dev_$(PACKAGE)_sandbox 2>/dev/null || true
	docker run --name dev_$(PACKAGE)_sandbox \
		$(DOCKER_OPTS) \
		--network host \
		-dt $(IMAGE_SANDBOX) jupyter-lab --ip 0.0.0.0 notebooks
	@sleep 2
	@docker exec -it dev_$(PACKAGE)_sandbox jupyter server list
