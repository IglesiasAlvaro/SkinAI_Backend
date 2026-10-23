import io
import json
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="SkinAI v2 - API de Inferencia",
    description="Backend de análisis dermatológico experimental (Prototipo Académico)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de configuración
BASE_DIR = Path(__file__).parent
MODEL_PATH = 'efficientnetb0_skinai_v2_finetuned.keras'
MAPPING_PATH = 'label_mapping.json'
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # Limitación estricta de 10 MB

# Variables globales de la APP
model = None
classes_order = []

@app.on_event("startup")
def startup_event():
    global model, classes_order
    # 1. Cargar el modelo una sola vez al iniciar
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"No se encontró el archivo del modelo en la ruta: {MODEL_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✔ [STARTUP] Modelo SkinAI v2 cargado exitosamente.")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo TensorFlow/Keras: {e}")

    # 2. Cargar el mapeo de clases oficial para garantizar el alineamiento correcto
    if not os.path.exists(MAPPING_PATH):
        raise RuntimeError(f"No se encontró el mapeo de clases en: {MAPPING_PATH}")

    with open(MAPPING_PATH, 'r') as f:
        mapping_data = json.load(f)

    # Construimos el array ordenado por su índice correspondiente (0 a 6)
    classes_order = [mapping_data['index_to_label'][str(i)] for i in range(7)]
    print(f"✔ [STARTUP] Mapeo de clases cargado: {classes_order}")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Endpoint para verificar el estado de salud de la API y comprobar si el modelo está en memoria.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El servidor se encuentra activo pero el modelo aún no se ha inicializado en memoria."
        )
    return {
        "status": "healthy",
        "model_version": "SkinAI_v2_finetuned",
        "framework": f"TensorFlow {tf.__version__}"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint de predicción clínica simulada. Recibe una imagen, aplica el procesamiento requerido
    por EfficientNetB0 (redimensión a 224x224, sin división de escala manual ya que el modelo la incluye internamente)
    y devuelve las probabilidades ordenadas.
    """
    # 1. Verificar tamaño de archivo subido sin guardarlo en disco
    # Para evitar cargar imágenes masivas que desborden la memoria de la API
    contents = await file.read()
    file_size = len(contents)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo es demasiado grande ({file_size / (1024*1024):.2f} MB). El límite máximo es de 10.0 MB."
        )

    # 2. Intentar parsear y validar la estructura de la imagen
    try:
        image = Image.open(io.BytesIO(contents))
        # Convertimos forzosamente a RGB para evitar problemas con canales alfa (PNG) o escala de grises
        image = image.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo enviado no pudo ser identificado como una imagen válida o compatible."
        )

    # 3. Preprocesamiento (Redimensionar a 224x224)
    try:
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized, dtype=np.float32) # Conservar escala 0-255 conforme a los requisitos
        img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de lote (Batch size: 1)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento de redimensión de la imagen: {str(e)}"
        )

    # 4. Inferencia con el modelo cargado
    try:
        predictions = model.predict(img_array, verbose=0)[0] # Obtenemos el vector de probabilidades
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error inesperado al ejecutar la inferencia del modelo: {str(e)}"
        )

    # 5. Estructurar y ordenar los resultados de mayor a menor
    scores = {classes_order[i]: float(predictions[i]) for i in range(len(classes_order))}
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return {
        "success": True,
        "model_version": "SkinAI v2 (EfficientNetB0 Fine-Tuned)",
        "class_scores": sorted_scores,
        "disclaimer": (
            "AVISO IMPORTANTE DE SEGURIDAD: Este sistema corresponde a un prototipo académico experimental "
            "diseñado exclusivamente con fines de investigación científica y evaluación técnica. No es un dispositivo "
            "médico, no emite un diagnóstico clínico definitivo, ni debe sustituir en ningún caso el criterio, "
            "evaluación y examen presencial de un dermatólogo u otro profesional cualificado de la salud."
        )
    }
