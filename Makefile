.PHONY: install train serve test lint clean

install:
	pip install -e ".[dev]"

train:
	python -c "from src.pipeline import FraudDetectionPipeline; FraudDetectionPipeline().run_training()"

train-file:
	python -c "from src.pipeline import FraudDetectionPipeline; FraudDetectionPipeline().run_training(data_file='$(FILE)')"

serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x -q

lint:
	python -m py_compile src/data/ingestion.py src/data/preprocessing.py \
	    src/features/build_features.py src/models/train.py \
	    src/models/evaluate.py src/models/predict.py \
	    src/pipeline.py api/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f data/models/*.pkl data/processed/*.pkl data/models/*.json
	rm -rf .pytest_cache htmlcov .coverage
