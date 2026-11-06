.PHONY: sync lint format test serve docker-build docker-serve docker-train clean

sync:
	uv sync --dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

train:
	uv run scripts/train.py

serve:
	uv run churn-gradio

docker-build:
	docker build -t churn-pred:latest .

# app en ligne locale : le bundle du modele est monte en volume
docker-serve:
	docker run --rm -p 7860:7860 -v $(CURDIR)/models:/app/models churn-pred:latest

# entrainement dans le conteneur : donnees brutes montees en volume
docker-train:
	docker run --rm -v $(CURDIR)/data:/app/data -v $(CURDIR)/models:/app/models -v $(CURDIR)/artifacts:/app/artifacts churn-pred:latest python scripts/train.py

clean:
	rm -rf .pytest_cache .ruff_cache artifacts/* models/*
