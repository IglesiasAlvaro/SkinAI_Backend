import os
import re
import math
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

# 1. Cargar variables de entorno
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Usar la clave service_role / sb_secret_...
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicializar clientes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

KNOWLEDGE_DIR = "knowledge"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # Debe coincidir con vector(768) en Supabase


def extract_metadata_and_content(file_content):
    """Separa el encabezado entre '---' (solo la primera pareja) del cuerpo."""
    parts = re.split(r"^---\s*$", file_content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) >= 3:
        metadata_raw = parts[1].strip()
        body = parts[2].strip()
        metadata = {}
        for line in metadata_raw.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        return metadata, body
    return {}, file_content


def chunk_text(text, max_chars=800):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
        else:
            current_chunk += p + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def normalize(vector):
    """Normaliza el vector (necesario con output_dimensionality != 3072)."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed_chunk(chunk, retries=3):
    """Genera el embedding de un fragmento con reintentos ante errores temporales."""
    for attempt in range(1, retries + 1):
        try:
            response = ai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return normalize(response.embeddings[0].values)
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 * attempt
            print(f"  ⚠ Intento {attempt} falló ({e}). Reintentando en {wait}s...")
            time.sleep(wait)


def process_and_upload():
    print("Iniciando procesamiento con el nuevo SDK de Gemini...")
    files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".md")]

    saved = 0
    failed = 0

    for filename in sorted(files):
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, body = extract_metadata_and_content(content)
        metadata["filename"] = filename

        chunks = chunk_text(body)
        print(f"\nProcesando {filename} ({len(chunks)} fragmentos)...")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            try:
                embedding = embed_chunk(chunk)

                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i + 1

                data = {
                    "content": chunk,
                    "metadata": chunk_metadata,
                    "embedding": embedding,
                }

                supabase.table("skinai_knowledge").insert(data).execute()
                saved += 1
                print(f"  ✔ Fragmento {i + 1} guardado correctamente.")
                time.sleep(0.3)  # evita límites de tasa en la capa gratuita
            except Exception as e:
                failed += 1
                print(f"  ❌ Error al guardar fragmento {i + 1}: {e}")

    return saved, failed


if __name__ == "__main__":
    saved, failed = process_and_upload()
    print(f"\nResumen: {saved} fragmentos guardados, {failed} con error.")
    if failed == 0:
        print("¡Base de datos vectorial poblada exitosamente!")
    else:
        print("Hubo errores: revisa los mensajes ❌ de arriba antes de continuar.")
import os
import math

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pregunta = "¿Qué es el melanoma y cuáles son sus señales de alerta?"

resp = ai_client.models.embed_content(
    model="gemini-embedding-001",
    contents=pregunta,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    ),
)
vec = resp.embeddings[0].values
norm = math.sqrt(sum(v * v for v in vec))
vec = [v / norm for v in vec]

result = supabase.rpc(
    "match_documents",
    {"query_embedding": vec, "match_count": 3},
).execute()

for row in result.data:
    similitud = round(row["similarity"], 3)
    archivo = row["metadata"].get("filename")
    texto = row["content"][:150].replace("\n", " ")
    print(f"{similitud} | {archivo} | {texto}")