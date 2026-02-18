ENV_FILE := .env
UID := $$(id -u)
GID := $$(id -g)
GITCONFIG := $${HOME}/.gitconfig
GITCREDENTIALS := $${HOME}/.git-credentials
DATE_STRING := $(shell date "+%Y-%m-%d")
JOB := canoe$(date +%Y%m%d_%H%M%S)
HOST := $${HOSTNAME}

.PHONY: help env up down ps start build deploy finish log log1 log2 status status1 status2 mint upload node resource

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
		echo "REGISTRY=${HOST}:5000" >> $(ENV_FILE); \
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
		bash -c 'git config --global --add safe.directory /canoe; source /opt/venv/bin/activate; exec bash'

build: env ## Build (or rebuild) the 'dev' container and start it
	@docker compose up -d --build dev

deploy: env ## Deploy a multi-node job to cluster
	@docker tag canoe:latest ${HOST}:5000/canoe:tmp
	@docker push ${HOST}:5000/canoe:tmp
	@USER_UID="$$(id -u)" USER_GID="$$(id -g)" REGISTRY="${HOST}:5000" docker stack deploy -c deploy.yaml ${JOB}
	@echo -e "\033[32m[OK]\033[0m ${JOB} deployed"

finish: ## Clean up the deployed job
	@docker stack rm ${JOB}

log: ## Show the job log file
	docker service logs ${JOB}_captain

log1: ## Show the job log file for crew1
	docker service logs ${JOB}_crew1

log2: ## Show the job log file for crew2
	docker service logs ${JOB}_crew2

status: ## Show the job status
	docker service ps ${JOB}_captain --no-trunc

status1: ## Show the job status for crew1
	docker service ps ${JOB}_crew1 --no-trunc

status2: ## Show the job status for crew2
	docker service ps ${JOB}_crew2 --no-trunc

mint: ## Mint the current environment to docker hub
	docker exec canoe-dev-1 bash -lc 'rm -f /etc/git-credentials /root/.git-credentials /home/*/.git-credentials 2>/dev/null || true'
	docker tag canoe:latest docker.io/luminoctum/ubuntu22.04-cuda12.9-py3.10-canoe:${DATE_STRING}

save: ## Save a temporary version
	docker save canoe:latest | gzip > ${HOME}/data/canoe_tmp.tar.gz

load: ## Load a temporary version
	gunzip -c ${HOME}/data/canoe_tmp.tar.gz | docker load

upload: ## Upload the minted image to docker hub
	# Refuse to push if the image still contains git credential files
	docker run --rm docker.io/luminoctum/ubuntu22.04-cuda12.9-py3.10-canoe:${DATE_STRING} sh -c '\
		if ls /home/*/.git-credentials >/dev/null 2>&1; then \
			echo "Refusing to push image: git credential files detected inside the image." >&2; \
			exit 1; \
		fi \
	'
	docker push docker.io/luminoctum/ubuntu22.04-cuda12.9-py3.10-canoe:${DATE_STRING}

node: ## Show nodes in cluster
	docker node ls

resource: ## Print out cluster resource
	@printf "%-35s %-10s %-15s %-10s\n" "NODE" "CPUs" "MEMORY (GB)" "GPUs"
	@printf "%-35s %-10s %-15s %-10s\n" "----" "----" "-----------" "----"
	@for node in $$(docker node ls --format "{{.Hostname}}"); do \
		RESOURCES=$$(docker node inspect $$node --format ' \
			{{.Description.Resources.NanoCPUs}} \
			{{.Description.Resources.MemoryBytes}} \
			{{if .Description.Resources.GenericResources}}{{range .Description.Resources.GenericResources}}{{if .NamedResourceSpec}}{{.NamedResourceSpec.Kind}} {{else if .DiscreteResourceSpec}}{{.DiscreteResourceSpec.Kind}} {{end}}{{end}}{{else}}0{{end}}'); \
		\
		CPUS=$$(echo $$RESOURCES | awk '{print $$1 / 1000000000}'); \
		MEM=$$(echo $$RESOURCES | awk '{print $$2 / 1024 / 1024 / 1024}'); \
		GPU_COUNT=$$(echo $$RESOURCES | grep -o "gpu" | wc -l); \
		\
		printf "%-35s %-10s %-15.2f %-10s\n" $$node $$CPUS $$MEM $$GPU_COUNT; \
	done
