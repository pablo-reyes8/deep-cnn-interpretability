FROM python:3.11.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace:/workspace/src"

COPY requirements-mlops.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-mlops.txt

COPY . .

CMD ["python3", "-m", "resnet101.src.training.train_mlflow"]

