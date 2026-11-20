
# ==========================================
# PRACTICA 3 - SESION 4
# Dockerfile para FastAPI + ONNX
# ==========================================

# 1. Imagen base requerida por la guia
FROM python:3.10-slim

# 2. Configuracion de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 3. Directorio de trabajo
WORKDIR /app

# 4. Copiar dependencias
COPY requirements-api.txt .
# 5. Instalar dependencias
RUN python -m pip install \
    --no-cache-dir \
    -r requirements-api.txt

# 6. Copiar codigo de la API
COPY app/ ./app/

# 7. Copiar modelo ONNX
COPY models/modelo.onnx ./models/modelo.onnx

# 8. Crear usuario sin privilegios
RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /app

USER apiuser

# 9. Exponer el puerto de FastAPI
EXPOSE 8000

# 10. Ejecutar servidor
CMD ["uvicorn", "app.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
