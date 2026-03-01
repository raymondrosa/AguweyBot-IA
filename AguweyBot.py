# ============================================
# AGUWEYBOT PRO - RAG + VISUAL
# ============================================

import os
import base64
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

MODEL_NAME = "phi3:mini"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "vector_db"

SYSTEM_PROMPT = """
Eres AguweyBot PRO.

Eres un asistente profesional especializado en ingeniería y ciencias aplicadas.
Tu audiencia principal son ingenieros, investigadores y profesionales técnicos.

Lineamientos de comportamiento:

1. Mantén un tono amable, respetuoso y profesional en todo momento.
2. Responde con enfoque científico e ingenieril.
3. Explica con claridad técnica, precisión conceptual y estructura lógica.
4. Cuando sea necesario, utiliza terminología técnica apropiada.
5. Fundamenta tus respuestas en principios verificables.
6. Si la información disponible no es suficiente, indícalo con honestidad.
7. No inventes datos ni supuestos.
8. Mantén empatía profesional: comprende que el usuario puede estar resolviendo problemas reales.
9. Estructura tus respuestas en secciones cuando sea útil.
10. Prioriza el pensamiento paso a paso antes de concluir.

Reglas críticas:

- Usa exclusivamente el contexto técnico proporcionado por el sistema.
- Si el contexto no contiene la información necesaria, indícalo claramente.
- No extrapoles más allá del contexto recuperado.
- Mantén coherencia científica y consistencia lógica.

Tu objetivo es asistir a profesionales de la ingeniería al nivel de un experto técnico.
"""
# ==========================
# FONDO PERSONALIZADO
# ==========================

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}

        .main .block-container {{
            background-color: rgba(0,0,0,0.85);
            padding: 2rem;
            border-radius: 20px;
        }}

        h1, h2, h3 {{
            color: #00ffe0 !important;
        }}

        p, label {{
            color: #ffffff !important;
        }}

        </style>
        """, unsafe_allow_html=True)


# ==========================
# CARGAR MODELO
# ==========================

@st.cache_resource
def cargar_llm():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        num_ctx=4096,
        top_p=0.9,
        repeat_penalty=1.1
    )


@st.cache_resource
def cargar_retriever():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ==========================
# INTERFAZ
# ==========================

st.set_page_config(page_title="AguweyBot PRO", page_icon="⚡")

set_background("fondo.png")

with st.sidebar:
    st.markdown("## 🤖 AGUWEYBOT PRO")

    if os.path.exists("logo.png"):
        st.image("logo.png", width=220)
    else:
        st.info("Coloca logo.png en la carpeta")

    st.markdown("---")
    st.caption("Arquitectura RAG Profesional")

st.title("⚡ AguweyBot PRO")
st.caption("Sistema cognitivo con recuperación semántica")

llm = cargar_llm()
retriever = cargar_retriever()

pregunta = st.text_input("Escribe tu pregunta:")

if pregunta:

    with st.spinner("Analizando..."):

        docs = retriever.invoke(pregunta)

        if not docs:

            mensajes = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=pregunta)
            ]

            respuesta = llm.invoke(mensajes)

            st.markdown("### Respuesta")
            st.write(respuesta.content)

        else:

            contexto_rag = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt_final = f"""
Contexto técnico relevante:
{contexto_rag}

Pregunta del usuario:
{pregunta}
"""

            mensajes = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_final)
            ]

            respuesta = llm.invoke(mensajes)

            st.markdown("### Respuesta")
            st.write(respuesta.content)
            prompt_final = f"""
Contexto técnico relevante:
{contexto_rag}

Instrucciones:
- Analiza cuidadosamente el contexto.
- Extrae únicamente la información pertinente.
- Responde con rigor técnico.
- Si el contexto no contiene la respuesta, indícalo.

Pregunta del usuario:
{pregunta}
"""

            mensajes = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt_final)
            ]

            respuesta = llm.invoke(mensajes)

            st.markdown("### Respuesta")
            st.write(respuesta.content)