.PHONY: all lint test train validate docker \
        preprod-up preprod-down preprod-logs preprod-ps \
        smoke deploy rollback clean versions help

COMPOSE_FILE = docker-compose.preprod.yml
VERSION ?= latest

all: lint test train validate docker
	@echo "✓ Pipeline CI/CD local completado."

lint:
	@echo "=== Linting con flake8 ==="
	flake8 src/ tests/ api/ --config=setup.cfg
	@echo "Linting OK"

test:
	@echo "=== Tests unitarios ==="
	python3 -m pytest tests/ --ignore=tests/smoke -v --tb=short --cov=src --cov-report=term-missing

train:
	@echo "=== Entrenando modelo ==="
	python src/generate_data.py
	python src/train_pipeline.py

validate:
	@echo "=== Validando métricas ==="
	python src/validate_model.py

docker:
	@echo "=== Build imagen API ==="
	docker build -t renovacion-prestamo-api:local .

preprod-up:
	@echo "=== Levantando preproducción ==="
	docker compose -f $(COMPOSE_FILE) up --build -d
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "MLflow: http://localhost:5000"

preprod-down:
	docker compose -f $(COMPOSE_FILE) down -v
	@echo "Entorno detenido."

preprod-logs:
	docker compose -f $(COMPOSE_FILE) logs -f

preprod-ps:
	docker compose -f $(COMPOSE_FILE) ps

smoke:
	@echo "=== Smoke tests ==="
	python3 -m pytest tests/smoke/ -v --tb=short

deploy:
	bash deploy.sh $(VERSION)

rollback:
	@echo "=== Rollback a versión $(VERSION) ==="
	docker compose -f $(COMPOSE_FILE) down
	docker tag renovacion-prestamo-api:$(VERSION) renovacion-prestamo-api:latest
	docker compose -f $(COMPOSE_FILE) up -d

versions:
	python src/manage_versions.py

clean:
	rm -rf artifacts/ mlruns/ mlartifacts/ __pycache__ .coverage htmlcov/ .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Limpieza completada."

help:
	@echo "make all           — lint + test + train + validate + docker"
	@echo "make test          — tests unitarios"
	@echo "make train         — entrenar modelo"
	@echo "make validate      — quality gate"
	@echo "make preprod-up    — levantar stack"
	@echo "make smoke         — smoke tests"
	@echo "make deploy        — despliegue manual"
	@echo "make versions      — versiones MLflow"