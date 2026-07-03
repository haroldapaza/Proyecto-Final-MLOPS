"""api/app.py — API REST para predicción de renovación de préstamo."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import predictor
from api.schemas import HealthResponse, PrediccionOutput, RenovacionInput

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | API | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar la API."""
    log.info("Iniciando API en entorno: %s", os.getenv("ENV", "dev"))
    predictor.cargar()
    log.info("Modelo listo: %s", type(predictor.modelo).__name__)
    yield
    log.info("API cerrando")


app = FastAPI(
    title="API Renovación de Préstamo — Preproducción",
    description="Modelo de propensión para identificar clientes con mayor probabilidad de aceptar renovación de préstamo.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    debug=os.getenv("DEBUG", "False").lower() == "true",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    return {
        "api": "Renovación de Préstamo",
        "env": os.getenv("ENV", "dev"),
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    return HealthResponse(
        status="ok",
        modelo=type(predictor.modelo).__name__,
        version=os.getenv("MODEL_VERSION", "1.0.0"),
        recall=float(predictor.metricas.get("recall", 0)),
        env=os.getenv("ENV", "dev"),
    )


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(cliente: RenovacionInput):
    try:
        datos = cliente.model_dump()
        resultado = predictor.predecir(datos)
        return PrediccionOutput(**resultado)
    except ValueError as e:
        log.warning("Entrada inválida: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("Error inesperado en predicción")
        raise HTTPException(status_code=500, detail="Error interno en predicción")