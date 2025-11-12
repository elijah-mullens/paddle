ENV_FILE := .env
UID := $$(id -u)
GID := $$(id -g)

.PHONY: env up down build

env:
	@echo "USER=$$(id -un)" > $(ENV_FILE)
	@echo "USER_UID=$$(id -u)" >> $(ENV_FILE)
	@echo "USER_GID=$$(id -g)" >> $(ENV_FILE)
	@echo "Wrote $(ENV_FILE):"; cat $(ENV_FILE)

up: env
	@docker compose up -d

down:
	@docker compose down

ps:
	@docker compose ps

start:
	@docker compose exec --user $(UID):$(GID) dev bash

build: env
	@docker compose up -d --build dev
