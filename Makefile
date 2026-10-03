.PHONY: install lint test train clean

install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest tests/ -v

train:
	python src/train.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
