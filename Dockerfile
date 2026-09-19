FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Instalar dependencias básicas del sistema para compilación y comprobaciones de salud
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY . .

# Exponer el puerto de la API FastAPI
EXPOSE 8000

# Comando de arranque por defecto en modo desarrollo con recarga automática
CMD ["uvicorn", "src.agent_service.api.webhook:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
