ENV_FILE := .env
UID := $$(id -u)
GID := $$(id -g)
GITCONFIG := $${HOME}/.gitconfig
GITCREDENTIALS := $${HOME}/.git-credentials
JOB := canoe$(date +%Y%m%d_%H%M%S)

.PHONY: help env up down ps start build canoe return log crew1

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

run: env
	@USER_UID="$$(id -u)" USER_GID="$$(id -g)" docker stack deploy -c canoe.yaml ${JOB}
	@docker service scale ${JOB}_captain=1 ${JOB}_crew1=1
	@echo "${JOB} dispatched"
	#@docker stack rm ${JOB}
	#@echo "${JOB} returned"

return:
	@docker stack rm ${JOB}

log:
	docker service logs -f canoe_captain

crew1:
	docker service logs -f canoe_crew1
