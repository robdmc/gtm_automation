#! /usr/bin/make 


.PHONY: help \
build_local \
build_osx \
build_linux \
build_all \
shell \
app \
app_daemon\
tasks\
tasks_daemon\
down\
prune

help:  ## Print the help documentation
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


build_local: ## Build images locally and don't send to dockerhub
	docker build -t robdmc/gtm .

build_osx: ## Build images for osx and push to docker hub
	docker buildx build --platform linux/arm64 --push -t robdmc/gtm .

build_linux: ## Build images for linux and osx and push to docker hub
	docker buildx build --platform linux/amd64 --push -t robdmc/gtm .

build_all: ## Build images for linux and osx and push to docker hub
	docker buildx build --platform linux/amd64,linux/arm64 --push -t robdmc/gtm .

shell: ## Open a shell in the container
	docker-compose run --rm shell

app: ## start streamlit app
	docker-compose up app

restart_app: ## Restart the app to reset cache
	-docker-compose stop app
	docker-compose up -d app



tasks: ## start streamlit app
	docker-compose run --rm tasks

tasks_daemon: ## start streamlit app in daemon mode
	docker-compose up -d tasks_daemon

app_daemon: ## start streamlit app as daemon
	docker-compose up -d app

daemon_all: app_daemon tasks_daemon ## Start all daemon services

down: ## stop all docker-compose services
	docker-compose down

nuke: ## Completely reset docker
	docker-compose down
	-docker images -q | xargs docker rmi -f
	-docker ps -aq | xargs docker rm
	-docker system prune --all

