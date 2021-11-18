# variables
IMAGE=jumanji
DOCKER_RUN_FLAGS=--rm --volume $(PWD):/home/app/jumanji
DOCKER_RUN=docker run $(DOCKER_RUN_FLAGS) $(IMAGE)
TEST=jumanji examples

# Makefile

build:
	docker build . -f Dockerfile -t $(IMAGE)

style: clean build
	$(DOCKER_RUN) pre-commit run --all-files

test: build
	$(DOCKER_RUN) pytest --verbose $(TEST)

bash: build
	docker run -it $(DOCKER_RUN_FLAGS) $(IMAGE) bash
