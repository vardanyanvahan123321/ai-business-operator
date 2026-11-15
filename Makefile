.DEFAULT_GOAL := help
.PHONY: help demo demo-traffic demo-logs demo-down demo-reset test-infra test-infra-down test lint format evals screenshots

COMPOSE := docker compose
TEST_DB := postgresql+asyncpg://operator:operator@localhost:55432/operator

help:  ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

demo:  ## Start the full demo (no API keys needed)
	$(COMPOSE) up --build -d --wait
	@echo ""
	@echo "  Admin panel    http://localhost:8088/panel    (password: demo)"
	@echo "  Customer form  http://localhost:8088/request"
	@echo "  Mail inbox     http://localhost:8025          (all email stays local)"
	@echo "  Competitor     http://localhost:8090/autofix/prices"
	@echo ""
	@echo "  Try: make demo-traffic   - sends three customer emails through the pipeline"

demo-traffic:  ## Send a few live customer emails into the running demo
	$(COMPOSE) exec web python -m ai_operator.demo traffic

demo-logs:  ## Follow the logs of the app containers
	$(COMPOSE) logs -f web worker scheduler

demo-down:  ## Stop the demo (data is kept)
	$(COMPOSE) down

demo-reset:  ## Stop the demo and delete its data
	$(COMPOSE) down -v

test-infra:  ## Start Postgres (pgvector) and Redis for the tests
	@docker start abo-test-pg abo-test-redis 2>/dev/null || ( \
	  docker run -d --name abo-test-pg -e POSTGRES_USER=operator -e POSTGRES_PASSWORD=operator \
	    -e POSTGRES_DB=operator -p 127.0.0.1:55432:5432 pgvector/pgvector:pg17 && \
	  docker run -d --name abo-test-redis -p 127.0.0.1:56379:6379 redis:7-alpine )

test-infra-down:  ## Stop the test containers
	docker stop abo-test-pg abo-test-redis

test:  ## Run the test suite with coverage (needs: make test-infra)
	uv run pytest --cov

lint:  ## Ruff + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:  ## Auto-format the code
	uv run ruff check --fix .
	uv run ruff format .

evals:  ## Run the AI quality evals (PROVIDER=mock|gemini|anthropic|openai)
	DATABASE_URL=$(TEST_DB) uv run python -m evals.run --provider $(or $(PROVIDER),mock)

screenshots:  ## Re-take README screenshots from the running demo
	uv run --group screenshots python scripts/screenshots.py
