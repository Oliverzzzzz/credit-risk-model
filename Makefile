.PHONY: install train api notebook mlflow clean

install:
	pip install -r requirements.txt

train:
	python pipeline.py

api:
	uvicorn src.api.main:app --reload --port 8000

notebook:
	jupyter notebook notebooks/

mlflow:
	mlflow ui --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
