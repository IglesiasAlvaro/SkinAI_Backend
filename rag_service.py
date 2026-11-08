import os
import math

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role / sb_secret_... (solo backend)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Modelo de generación: cámbialo desde el .env si Google ajusta el plan gratuito
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

TOP_K = 4              # fragmentos que se pasan como contexto
MIN_SIMILARITY = 0.55  # por debajo de esto se considera "sin información"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = """Eres el asistente informativo de SkinAI, un sistema que clasifica lesiones de la piel.
Respondes preguntas sobre 7 tipos de lesiones (akiec, bcc, bkl, df, mel, nv, vasc), los signos de alerta y las limitaciones del sistema.

Reglas:
1. Responde SOLO con la información del CONTEXTO que se te entrega. No inventes datos médicos ni uses conocimiento externo.
2. Si el contexto no contiene la respuesta, di claramente que no tienes información suficiente sobre eso y sugiere consultar a un dermatólogo.
3. Nunca des un diagnóstico. No digas "tienes X"; explica información general.
4. Responde en español, con lenguaje claro y en un máximo de 150 palabras.
5. Si la pregunta sugiere una lesión preocupante (cambios, sangrado, crecimiento rápido), recomienda acudir a un profesional de la salud.
6. Termina con esta nota en una línea: "Esta información es orientativa y no reemplaza una consulta médica."
"""

FALLBACK_ANSWER = (
    "No tengo información suficiente en mi base de conocimiento para responder esa pregunta. "
    "Te recomiendo consultar a un dermatólogo. "
    "Esta información es orientativa y no reemplaza una consulta médica."
)


def _normalize(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def embed_query(question: str):
    response = ai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return _normalize(response.embeddings[0].values)


def retrieve(question: str, k: int = TOP_K):
    result = supabase.rpc(
        "match_documents",
        {"query_embedding": embed_query(question), "match_count": k},
    ).execute()
    return result.data or []


def build_prompt(question: str, rows):
    context = "\n\n---\n\n".join(
        f"[Fuente: {r['metadata'].get('filename', 'desconocida')}]\n{r['content']}"
        for r in rows
    )
    return f"CONTEXTO:\n{context}\n\nPREGUNTA DEL USUARIO:\n{question}"


def answer_question(question: str) -> dict:
    rows = retrieve(question)

    # Sin resultados relevantes: no se gasta cuota llamando al modelo
    if not rows or rows[0]["similarity"] < MIN_SIMILARITY:
        return {"answer": FALLBACK_ANSWER, "sources": []}

    response = ai_client.models.generate_content(
        model=CHAT_MODEL,
        contents=build_prompt(question, rows),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )

    sources = sorted({r["metadata"].get("filename", "") for r in rows if r["metadata"]})
    return {"answer": (response.text or FALLBACK_ANSWER).strip(), "sources": sources}


if __name__ == "__main__":
    # Prueba rápida desde la terminal: python rag_service.py
    pregunta = "¿Cuáles son las señales de alerta del melanoma?"
    print(answer_question(pregunta))