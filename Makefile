ENV_FILE := .env
UID := $$(id -u)
GID := $$(id -g)
GITCONFIG := "$${HOME}/.gitconfig"

.PHONY: help env up down ps start build

# Show help for each target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env: ## Generate the .env file with user and gitconfig variables
	@echo "USER=$$(id -un)" > $(ENV_FILE)
	@echo "USER_UID=$$(id -u)" >> $(ENV_FILE)
	@echo "USER_GID=$$(id -g)" >> $(ENV_FILE)
	@if [ -f $(GITCONFIG) ]; then \
		echo "GITCONFIG=$(GITCONFIG)" >> $(ENV_FILE); \
	else \
		touch $(GITCONFIG); \
		echo "GITCONFIG=$(GITCONFIG)" >> $(ENV_FILE); \
	fi
	@echo "Wrote $(ENV_FILE):"; cat $(ENV_FILE)

up: env ## Start docker containers in the background
	@docker compose up -d

down: ## Stop and remove docker containers
	@docker compose down

ps: ## Show container status
	@docker compose ps

start: ## Open a bash shell inside the 'dev' container as the host user, exit without error
	@docker compose exec --user $(UID):$(GID) dev bash || true

build: env ## Build (or rebuild) the 'dev' container and start it
	@docker compose up -d --build dev
