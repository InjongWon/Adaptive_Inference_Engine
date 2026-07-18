.PHONY: install test lint gateway monitor smoke bench

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

gateway:
	./scripts/run_gateway.sh

monitor:
	docker compose up -d

smoke:
	python -m benchmark.load_generator --workload smoke

bench:
	./scripts/run_experiment_matrix.sh
