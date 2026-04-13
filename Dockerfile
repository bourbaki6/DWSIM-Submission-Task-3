FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements_app.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_app.txt

COPY . .

RUN mkdir -p data models plots results

EXPOSE 8501

CMD ["python", "run_pipeline.py"]