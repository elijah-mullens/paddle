ENV_FILE := .env
UID := $$(id -u)
GID := $$(id -g)
GITCONFIG := $${HOME}/.gitconfig
GITCREDENTIALS := $${HOME}/.git-credentials
DATE_STRING := $(shell date "+%Y-%m-%d")
JOB := canoe$(date +%Y%m%d_%H%M%S)

.PHONY: help env up down ps start build deploy finish log status mint upload crew1

# Show help for each target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env: ## Generate the .env file with git configs first, then user info
	@# Check if either variable is missing
	@if ! grep -q '^GITCONFIG=' $(ENV_FILE) 2>/dev/null || \
	    ! grep -q '^GITCREDENTIALS=' $(ENV_FILE) 2>/dev/null; then \
		if [ -f "$${HOME}/.gitconfig" ]; then \
			echo "GITCONFIG=$${HOME}/.gitconfig" > $(ENV_FILE); \
		else \
			touch "$${HOME}/.gitconfig"; \
			echo "GITCONFIG=$${HOME}/.gitconfig" > $(ENV_FILE); \
		fi; \
		if [ -f "$${HOME}/.git-credentials" ]; then \
			echo "GITCREDENTIALS=$${HOME}/.git-credentials" >> $(ENV_FILE); \
		else \
			touch "$${HOME}/.git-credentials"; \
			echo "[credential]" >> "$${HOME}/.gitconfig"; \
			echo "    helper = store" >> "$${HOME}/.gitconfig"; \
			echo "GITCREDENTIALS=$${HOME}/.git-credentials" >> $(ENV_FILE); \
		fi; \
		echo "USER=$$(id -un)"    >> $(ENV_FILE); \
		echo "USER_UID=$$(id -u)" >> $(ENV_FILE); \
		echo "USER_GID=$$(id -g)" >> $(ENV_FILE); \
		echo "Created $(ENV_FILE):"; cat $(ENV_FILE); \
	fi

up: env ## Start docker containers in the background
	@docker compose up -d
	@docker compose exec dev bash -c '\
		USER_HOME=$$(eval echo ~$$USERNAME); \
		if [ -f /etc/git-credentials ] && [ ! -f $$USER_HOME/.git-credentials ]; then \
			cp /etc/git-credentials $$USER_HOME/.git-credentials; \
			chown $${USER_UID}:$${USER_GID} $$USER_HOME/.git-credentials; \
			git config --global credential.helper store; \
		fi \
	'

down: env ## Stop and remove docker containers
	@docker compose down

ps: env ## Show container status
	@docker compose ps

start: env ## Open a bash shell inside the 'dev' container as the host user, exit without error
	@docker compose exec --user $(UID):$(GID) dev \
		bash -c 'git config --global --add safe.directory /paddle; exec bash; source /opt/venv/bin/activate'

build: env ## Build (or rebuild) the 'dev' container and start it
	@docker compose up -d --build dev

deploy: env ## Deploy a program to cluster
	@USER_UID="$$(id -u)" USER_GID="$$(id -g)" docker stack deploy -c deploy.yaml ${JOB}
	@echo -e "\033[32m[OK]\033[0m ${JOB} deployed"

finish: ## Clean up the deployed job
	@docker stack rm ${JOB}

log: ## show the job log file
	docker service logs canoe_captain

status: ## show the job status
	docker service ps canoe_captain

restart: ## restart a job
	docker service scale ${JOB}_captain=1 ${JOB}_crew1=1

crew1:
	docker service logs -f canoe_crew1

mint: ## Mint the current environment
	docker commit paddle-dev-1 ubuntu22.04-cuda12.9-py3.10-canoe:latest
	docker tag ubuntu22.04-cuda12.9-py3.10-canoe:latest docker.io/luminoctum/ubuntu22.04-cuda12.9-py3.10-canoe:${DATE_STRING}

upload: ## Upload the minted image to docker hub
	docker push docker.io/luminoctum/ubuntu22.04-cuda12.9-py3.10-canoe:${DATE_STRING}
