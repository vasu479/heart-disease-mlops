# Heart Disease Risk Prediction API — serving image
# Build:  docker build -t heart-disease-api:latest .
# Run:    docker run -p 8000:8000 heart-disease-api:latest
# Test:   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
#           -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
#
# NOTE: this bakes the already-trained model into the image (models/model.joblib
# + models/metadata.json). Run `python src/train.py` locally BEFORE building so
# those files exist — the assignment's "container must build and run locally
# with sample input" check depends on the model already being present.

FROM python:3.12-slim

WORKDIR /app

# System deps: none beyond what python:slim ships — xgboost's wheel is manylinux
# and needs libgomp1 at runtime for OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
