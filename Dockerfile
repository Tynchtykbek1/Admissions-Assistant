FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt && \
    python -c "import torch; assert torch.version.cuda is None, 'CUDA-enabled PyTorch was installed'"

RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /data /app/uploads "$HF_HOME" && \
    chown -R appuser:appuser /app /data /home/appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
