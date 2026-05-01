.PHONY: install train api notebook mlflow clean

install:
	pip3 install -r requirements.txt

train:
	python3 pipeline.py

api:
	uvicorn src.api.main:app --reload --port 8000

notebook:
	jupyter notebook notebooks/

mlflow:
	mlflow ui --port 5000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
