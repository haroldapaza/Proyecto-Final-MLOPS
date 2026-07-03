#!/bin/bash

set -e

VERSION=${1:-latest}
IMAGE_NAME="renovacion-prestamo-api"
COMPOSE_FILE="docker-compose.preprod.yml"

echo "============================================="
echo " DEPLOY PREPRODUCCIÓN — versión: $VERSION"
echo "============================================="

echo "[1/6] Guardando versión actual..."
CURRENT=$(docker images --format "{{.Tag}}" $IMAGE_NAME 2>/dev/null | head -1 || echo "none")
echo "Versión actual: $CURRENT"

echo "[2/6] Construyendo imágenes..."
docker compose -f $COMPOSE_FILE build

echo "[3/6] Levantando entorno..."
docker compose -f $COMPOSE_FILE up -d

echo "[4/6] Ejecutando smoke tests..."
if python3 -m pytest tests/smoke/ -v --tb=short -q; then
    echo "Smoke tests OK"
else
    echo "Smoke tests fallaron. Ejecutando rollback..."
    docker compose -f $COMPOSE_FILE down

    if [ "$CURRENT" != "none" ]; then
        docker tag "$IMAGE_NAME:$CURRENT" "$IMAGE_NAME:latest"
        docker compose -f $COMPOSE_FILE up -d
    fi

    exit 1
fi

echo "[5/6] Tagging imagen..."
docker tag "$IMAGE_NAME:latest" "$IMAGE_NAME:$VERSION"

echo "[6/6] Verificación final..."
curl -sf http://localhost:8000/health | python3 -m json.tool

echo "============================================="
echo " DEPLOY EXITOSO — $IMAGE_NAME:$VERSION"
echo " API   : http://localhost:8000"
echo " Docs  : http://localhost:8000/docs"
echo " MLflow: http://localhost:5000"
echo "============================================="