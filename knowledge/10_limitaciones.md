---
titulo: Limitaciones de SkinAI y del análisis por imágenes
fuentes: Principios de IA médica general
---
# Limitaciones de SkinAI y del Análisis por IA

Es vital comprender qué es SkinAI y, más importante aún, qué NO es.

## 1. No es un Diagnóstico Médico
SkinAI es una herramienta basada en el dataset HAM10000 que analiza imágenes dermatoscópicas mediante Inteligencia Artificial y emite una probabilidad estadística. **Bajo ninguna circunstancia la salida del modelo constituye un diagnóstico clínico.** Solo un médico, preferiblemente un dermatólogo, puede diagnosticar una afección de la piel tras un examen físico y, si es necesario, una biopsia.

## 2. Contexto Clínico Ausente
SkinAI analiza solo los píxeles de una imagen bidimensional. La IA **no** sabe:
* La historia clínica del paciente (antecedentes familiares de cáncer).
* Si la lesión pica, duele, arde o ha sangrado recientemente.
* La textura física de la lesión al tacto.
* La ubicación de la lesión o hace cuánto tiempo apareció.

## 3. Limitación de Datos
El modelo solo conoce siete categorías (akiec, bcc, bkl, df, mel, nv, vasc). Existen cientos de otras enfermedades de la piel (infecciones, hongos, enfermedades autoinmunes, acné, etc.) que la red neuronal simplemente no está entrenada para reconocer.

Cualquier uso de esta herramienta debe ir acompañado del consejo estricto de **consultar a un profesional de la salud**.