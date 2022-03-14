.PHONY: build # in case we create file/dir with this name

# variables
IMAGE=jumanji
DOCKER_RUN_FLAGS=--rm --volume $(PWD):/home/app/jumanji
DISPLAY_FLAGS=--volume /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$(DISPLAY)
DOCKER_RUN=docker run $(DOCKER_RUN_FLAGS) $(IMAGE)
TEST=jumanji examples validation

# Makefile

build:
	docker build . -f Dockerfile -t $(IMAGE)

style: clean build
	$(DOCKER_RUN) pre-commit run --all-files

test: build
	$(DOCKER_RUN) pytest --verbose $(TEST)

bash: build
	docker run -it $(DOCKER_RUN_FLAGS) $(IMAGE) bash

bash_with_display: build
	docker run -it $(DOCKER_RUN_FLAGS) $(DISPLAY_FLAGS) $(IMAGE) bash
