.DEFAULT_GOAL := build

VERSION=$(shell python -c 'from asr_deepspeech import __version__; print(__version__)')

build:
	python -m build --wheel

test:
	python -m pytest tests/ -v

docker-vanilla:
	docker build -t zakuroai/asr_deepspeech:vanilla -f docker/vanilla/Dockerfile .

docker-sandbox: build
	docker build -t zakuroai/asr_deepspeech:sandbox -f docker/sandbox/Dockerfile .

docker: docker-vanilla docker-sandbox

clean:
	rm -rf dist/ build/ *.egg-info
