.PHONY: install check test lint pilot pilot-offline harvest clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e '.[pilot]'

check: lint test

lint:
	ruff check .

test:
	pytest

pilot:
	python scripts/run_pilot.py --mode auto

pilot-offline:
	python scripts/run_pilot.py --mode offline

harvest:
	noisevault harvest-ibm --max-backends 2

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
