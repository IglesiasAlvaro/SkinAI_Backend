from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chat_router import router as chat_router

app = FastAPI(
    title="SkinAI - API del Chatbot RAG",
    description="Servicio de chatbot informativo (sin TensorFlow, para ahorrar memoria)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "null",
        "https://iglesiasalvaro.github.io",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "skinai-chat"}