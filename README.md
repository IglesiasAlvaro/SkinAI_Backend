# SkinAI Backend 🩺

Backend de inferencia y asistente RAG para la clasificación e información de lesiones dermatológicas mediante visión por computadora e inteligencia artificial generativa.

> **Aviso Importante:** Este proyecto es un **prototipo académico y experimental**, y **NO es un dispositivo médico**. Sus predicciones y respuestas informativas no constituyen un diagnóstico clínico ni reemplazan la consulta, evaluación o criterio de un profesional de la salud o dermatólogo cualificado.

---

## 📋 Requisitos Previos

- Python 3.10 o superior (recomendado Python 3.11 o 3.12).
- Cuenta y proyecto en [Supabase](https://supabase.com) (con extensión `pgvector` y tabla/función vectorial).
- API Key de [Google AI Studio (Gemini)](https://aistudio.google.com).

---

## 🚀 Instalación

1. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/macOS:
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuración del Entorno (.env)

Copia el archivo `.env.example` para crear tu `.env` local:

```bash
# En Windows (PowerShell):
Copy-Item .env.example .env
# En Linux/macOS / Bash:
cp .env.example .env
```

Edita `.env` y completa tus credenciales y configuración:
- `SUPABASE_URL`: URL del proyecto en Supabase.
- `SUPABASE_KEY`: Clave de servicio (`service_role` o secreta) para operaciones backend.
- `GEMINI_API_KEY`: API Key de Google Gemini.
- `GEMINI_CHAT_MODEL`: Modelo para el chatbot (por defecto: `gemini-3.8-flash`).

---

## 📚 Poblar la Base de Conocimiento (RAG)

Para procesar y cargar los documentos clínicos desde la carpeta `knowledge/` hacia la base de datos vectorial de Supabase, ejecuta una sola vez:

```bash
python seed_knowledge.py
```

---

## 🖥️ Levantar el Servidor

Inicia la API en modo desarrollo con Uvicorn:

```bash
uvicorn main:app --reload
```

El servidor estará accesible en: `http://localhost:8000`  
Documentación interactiva Swagger: `http://localhost:8000/docs`

---

## 🔌 Endpoints Principales

- `GET /health`  
  Verifica el estado del servicio y si el modelo de red neuronal está cargado en memoria.

- `POST /predict`  
  Recibe una imagen dermatológica (`multipart/form-data`) y retorna la probabilidad para los 7 tipos de lesiones (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

- `POST /chat`  
  Recibe una pregunta en formato JSON (`{"question": "..."}`) y genera una respuesta orientativa fundamentada en la base de conocimiento utilizando RAG y Gemini.
