import { GoogleGenAI } from "@google/genai";
import { RewriteTone } from "../types";

const apiKey = process.env.API_KEY;

if (!apiKey) {
  console.error("API Key is missing. Application will not function correctly.");
}

const ai = new GoogleGenAI({ apiKey: apiKey || '' });

// Using gemini-3-pro-preview for complex text tasks (rewriting/nuance)
const MODEL_NAME = 'gemini-3-pro-preview';

export const rewriteTextStream = async (
  text: string, 
  tone: RewriteTone,
  onChunk: (text: string) => void
): Promise<void> => {
  
  const systemInstruction = `
    Actúa como un redactor académico de élite y especialista en evasión de detección de IA en Español Latinoamericano.

    OBJETIVO CRÍTICO:
    El texto resultante debe ser 100% INDETECTABLE por los siguientes detectores de Inteligencia Artificial:
    1. Copyleaks
    2. GPTZero
    3. ZeroGPT
    4. Originality.ai
    5. Scribbr
    6. Sidekicker
    7. Quillbot

    INSTRUCCIONES OBLIGATORIAS:
    
    1. LIMPIEZA VISUAL (FORMATO):
       - Elimina obsesivamente cualquier doble espacio, tabulación extraña o salto de línea injustificado.
       - El texto debe tener un espaciado tipográfico perfecto (un solo espacio entre palabras).

    2. ESTRATEGIA DE HUMANIZACIÓN (EVASIÓN):
       - ALTA PERPLEJIDAD: Usa palabras menos predecibles pero contextualmente perfectas.
       - ALTA BURSTINESS: Alterna drásticamente la longitud y estructura de las oraciones. Combina frases cortas y contundentes con oraciones compuestas complejas.
       - EVITA MARCADORES DE IA: No uses conectores sobreutilizados por LLMs (como "En conclusión", "Cabe destacar", "Por lo tanto", "Es importante resaltar") a menos que se sientan absolutamente orgánicos y necesarios.
       - NATURALIDAD: El texto debe sonar escrito por un humano educado, con matices y fluidez natural.

    3. CONTEXTO:
       - Tono: ${tone === RewriteTone.ACADEMIC ? 'Académico Riguroso (Nivel Tesis Doctoral)' : 'Formal Profesional (Nivel Corporativo/Ejecutivo)'}.
       - Idioma: Español Latinoamericano Neutro.

    SALIDA:
    - Devuelve SOLAMENTE el texto procesado.
    - NO incluyas Markdown de código (\`\`\`), ni saludos, ni notas.
  `;

  try {
    const responseStream = await ai.models.generateContentStream({
      model: MODEL_NAME,
      contents: [
        {
          role: 'user',
          parts: [{ text: text }]
        }
      ],
      config: {
        systemInstruction: systemInstruction,
        temperature: 1.0, // Increased for higher unpredictability/human-like variance
        topP: 0.95,
        topK: 64, // Increased vocabulary variety
      }
    });

    for await (const chunk of responseStream) {
      const chunkText = chunk.text;
      if (chunkText) {
        onChunk(chunkText);
      }
    }
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw error;
  }
};