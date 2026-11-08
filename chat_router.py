from fastapi import APIRouter, HTTPException
from google.genai import errors
from pydantic import BaseModel, Field

from rag_service import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return answer_question(req.question.strip())
    except errors.APIError as e:
        if e.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Se alcanzó el límite gratuito de consultas. Intenta de nuevo en un minuto.",
            )
        raise HTTPException(status_code=502, detail="Error al consultar el modelo de IA.")
    except Exception:
        raise HTTPException(status_code=500, detail="Error interno del servidor.")