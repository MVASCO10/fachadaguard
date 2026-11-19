# FachadaGuard — atalhos. A demo usa apenas Docker; avaliação, testes e PDF usam o venv local (make setup).
SHELL := /bin/bash
PY ?= .venv/bin/python
COMPOSE := docker compose
RUN := $(COMPOSE) run --rm
TOOL := $(RUN) --entrypoint python simulator

.PHONY: help env setup up down logs ps seed simulate-a simulate-b simulate-outage wait export reports demo \
        benchmark benchmark-large dataset evaluate test test-integration lint typecheck check

help: ## lista os comandos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

env:
	@test -f .env || (cp .env.example .env && echo ".env criado a partir de .env.example (valores de demonstração)")

setup: ## cria .venv local com dependências de desenvolvimento, dashboard e documentação
	python3.12 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dashboard,eval,dev]"

up: env ## sobe broker, banco, API, ingestor, worker e dashboard
	$(COMPOSE) up -d --build --wait

down: ## para os serviços (mantém volumes)
	$(COMPOSE) down

ps:
	$(COMPOSE) ps

logs: ## acompanha logs de API, ingestor e worker
	$(COMPOSE) logs -f --tail=100 api ingestor worker

seed: ## cadastra o edifício fictício de demonstração
	$(TOOL) scripts/demo_steps.py seed

simulate-a: ## missão simulada A (maio/2026)
	$(RUN) simulator mission --epoch A

simulate-b: ## missão simulada B (novembro/2026)
	$(RUN) simulator mission --epoch B

simulate-outage: ## missão B com perda de enlace, buffer local e reenvio
	$(RUN) simulator mission --epoch B --outage-at 0.35 --outage-seconds 40 --resend-last 15

wait: ## aguarda a fila de análises esvaziar (EXPECTED=n)
	$(TOOL) scripts/demo_steps.py wait --expected $${EXPECTED:-0}

export: ## exporta Postgres → Parquet particionado
	$(TOOL) scripts/demo_steps.py export

reports: ## gera relatório preliminar de cada missão
	$(TOOL) scripts/demo_steps.py reports

demo: up ## fluxo completo: missão A, análise, missão B com queda de enlace, comparação, lake e relatórios
	$(RUN) simulator mission --epoch A
	$(TOOL) scripts/demo_steps.py wait --expected 6
	$(RUN) simulator mission --epoch B --outage-at 0.35 --outage-seconds 40 --resend-last 15
	$(TOOL) scripts/demo_steps.py wait --expected 12
	$(TOOL) scripts/demo_steps.py export
	$(TOOL) scripts/demo_steps.py reports
	@echo "Dashboard: http://localhost:8501   API/OpenAPI: http://localhost:8000/docs"

benchmark: up ## 100 mil eventos de telemetria via MQTT + consultas DuckDB (artifacts/benchmark)
	$(TOOL) scripts/benchmark.py --events 100000
	@mkdir -p artifacts/benchmark && $(COMPOSE) cp worker:/app/artifacts/benchmark/. artifacts/benchmark/

benchmark-large: up ## opcional: 1 milhão de eventos
	$(TOOL) scripts/benchmark.py --events 1000000
	@mkdir -p artifacts/benchmark && $(COMPOSE) cp worker:/app/artifacts/benchmark/. artifacts/benchmark/

dataset: ## gera o dataset sintético em data/synthetic (manifesto + checksums)
	$(PY) scripts/generate_demo_dataset.py

evaluate: ## avaliação reproduzível do modelo (artifacts/evaluation)
	$(PY) scripts/evaluate.py

test: ## testes unitários e de integração local (SQLite) com cobertura
	$(PY) -m pytest --cov --cov-report=term-missing:skip-covered

test-integration: ## smoke test contra a stack Docker em execução
	$(PY) -m pytest -m integration tests/e2e -v

lint: ## ruff (lint + formatação)
	$(PY) -m ruff check .

typecheck: ## mypy no núcleo
	$(PY) -m mypy

check: lint typecheck test

