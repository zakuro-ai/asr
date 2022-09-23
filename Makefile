# PHONY are targets with no files to check, all in our case
.DEFAULT_GOAL := build

conf_file ?= .env
-include $(conf_file)

# Ensure that we have a configuration file
$(conf_file):
	$(error Please create a '$(conf_file)' file first, for example by copying example_conf.env. No '$(conf_file)' found)

cVERSION=$(shell python -c 'from $(PACKAGE) import __version__;print(__version__)')

# Makefile for launching common tasks
DOCKER_OPTS ?= \
	-v /dev/shm:/dev/shm \
	-v $(HOME)/.ssh:/home/foo/.ssh \
	-v $(HOME)/.config:/home/foo/.config \
	-v $(PWD):/workspace \
	-v $(SRV):/srv \
	-v $(FILESTORE):/FileStore \
	--privileged


help:
	@echo "Usage: make {build,  bash, ...}"
	@echo "Please check README.md for instructions"
	@echo ""


# BUILD:
build: build_wheels build_dockers

# BUILD DOCKER
build_dockers: build_docker_vanilla build_docker_sandbox 

build_docker_vanilla:
	docker build . -t $(IMAGE_VANILLA) --network host -f docker/vanilla/Dockerfile

build_docker_sandbox:
	docker build . -t  $(IMAGE_SANDBOX) --network host  -f docker/sandbox/Dockerfile

# BUILD WHEEL
build_wheels: build_wheel 

build_wheel:
	mv dist/*.whl dist/legacy/ && python setup.py bdist_wheel
	
	
# PUSH
push_dockers: push_docker_vanilla push_docker_sandbox

push_docker_sandbox:
	@docker tag $(IMAGE_SANDBOX) $(IMAGE_SANDBOX)-$(PACKAGE)_$(VERSION)
	docker push $(IMAGE_SANDBOX)
	docker push $(IMAGE_SANDBOX)-$(PACKAGE)_$(VERSION)

push_docker_vanilla:
	@docker tag $(IMAGE_VANILLA) $(IMAGE_VANILLA)-$(PACKAGE)_$(VERSION)
	docker push $(IMAGE_VANILLA)
	docker push $(IMAGE_VANILLA)-$(PACKAGE)_$(VERSION)

# PULL
pull: pull_docker_vanilla pull_docker_sandbox

pull_docker_vanilla:
	docker pull $(IMAGE_VANILLA)

pull_docker_sandbox:
	docker pull $(IMAGE_SANDBOX)

sandbox:
	@docker stop dev_$(PACKAGE)_sandbox || true
	@docker rm dev_$(PACKAGE)_sandbox || true
	docker run --name dev_$(PACKAGE)_sandbox \
		$(DOCKER_OPTS) \
		--network host \
		-dt $(IMAGE_SANDBOX) jupyter-lab --ip 0.0.0.0 notebooks 
	@sleep 2
	@docker exec -it dev_$(PACKAGE)_sandbox jupyter server list

# COMMON
tests:
	python -m $(PACKAGE).tests

