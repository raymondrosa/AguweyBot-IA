# ============================================
# AGUWEYBOT PRO - RAG + VISUAL + STREAMING (CORREGIDO)
# ============================================

import os
import base64
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
import time

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
# CALLBACK PARA STREAMING EN STREAMLIT
# ==========================

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler para streaming de tokens en Streamlit"""
    
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""
        self.first_token = True
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Procesa cada nuevo token"""
        self.text += token
        
        # Efecto de escritura con cursor parpadeante
        if self.first_token:
            self.container.markdown(f'<div class="respuesta-aguwey typing">{self.text}▌</div>', unsafe_allow_html=True)
            self.first_token = False
        else:
            self.container.markdown(f'<div class="respuesta-aguwey typing">{self.text}▌</div>', unsafe_allow_html=True)
        
        # Pequeña pausa para efecto visual
        time.sleep(0.01)

# ==========================
# FONDO PERSONALIZADO - CONTRASTE MÁXIMO
# ==========================

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(f"""
        <style>
        /* ===== ESTILOS ADICIONALES PARA STREAMING ===== */
        @keyframes blink {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        
        .respuesta-aguwey.typing {{
            border-right: 3px solid #00ffe0;
            animation: blink 1s infinite;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        /* ===== RESET COMPLETO ===== */
        html, body, .stApp, .main, .block-container, div, span, p, h1, h2, h3, h4, h5, h6, 
        label, .stTextInput label, .stMarkdown, .st-cx, .st-bx, .st-ci, .st-ck, .st-cw,
        .element-container, .row-widget, .stAlert, .stInfo, .stSuccess, .stWarning, .stError,
        .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak,
        .stButton p, .stSelectbox label, .stSlider label, .stDateInput label {{
            color: #ffffff !important;
        }}

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        .main .block-container {{
            background-color: rgba(0, 0, 0, 0.9) !important;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 255, 224, 0.2);
            border: 1px solid rgba(0, 255, 224, 0.3);
        }}

        h1, h2, h3 {{
            color: #00ffe0 !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            text-shadow: 0 0 10px rgba(0, 255, 224, 0.5);
        }}

        h4, h5, h6 {{
            color: #ffd966 !important;
            font-weight: 600 !important;
        }}

        .respuesta-aguwey {{
            background-color: #1a1a2a !important;
            border-left: 5px solid #00ffe0 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.7) !important;
            font-family: 'Segoe UI', system-ui, sans-serif !important;
            line-height: 1.6 !important;
            color: #ffffff !important;
        }}

        .respuesta-aguwey p {{
            color: #ffffff !important;
            line-height: 1.7 !important;
            font-size: 1.05rem !important;
            margin-bottom: 15px !important;
        }}

        .respuesta-aguwey strong {{
            color: #00ffe0 !important;
            font-weight: 700 !important;
        }}

        .respuesta-aguwey em {{
            color: #ffd966 !important;
            font-style: italic !important;
        }}

        .respuesta-aguwey code {{
            background-color: #2d2d3a !important;
            color: #ffb86b !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 0.95rem !important;
        }}

        .respuesta-aguwey pre {{
            background-color: #0a0a0f !important;
            border: 1px solid #00ffe0 !important;
            border-radius: 8px !important;
            padding: 15px !important;
            overflow-x: auto !important;
        }}

        .respuesta-aguwey pre code {{
            background-color: transparent !important;
            color: #f8f8f2 !important;
            padding: 0 !important;
        }}

        .respuesta-aguwey ul, .respuesta-aguwey ol {{
            color: #ffffff !important;
            margin: 10px 0 !important;
            padding-left: 25px !important;
        }}

        .respuesta-aguwey li {{
            color: #ffffff !important;
            margin: 5px 0 !important;
        }}

        .stTextInput label {{
            color: #ffffff !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin-bottom: 5px !important;
        }}

        .stTextInput input {{
            background-color: #2a2a3a !important;
            color: #ffffff !important;
            border: 2px solid #00ffe0 !important;
            border-radius: 10px !important;
            padding: 12px 15px !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
        }}

        .stTextInput input:focus {{
            border-color: #ffd966 !important;
            box-shadow: 0 0 15px rgba(0, 255, 224, 0.7) !important;
            background-color: #3a3a4a !important;
        }}

        .stTextInput input::placeholder {{
            color: #aaaaaa !important;
            font-style: italic !important;
        }}

        .css-1d391kg, .sidebar, [data-testid="stSidebar"] {{
            background-color: #0a0a0f !important;
            border-right: 2px solid #00ffe0 !important;
        }}

        [data-testid="stSidebar"] * {{
            color: #ffffff !important;
        }}

        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {{
            color: #00ffe0 !important;
        }}

        [data-testid="stSidebar"] .stAlert {{
            background-color: #1a1a2a !important;
            color: #ffffff !important;
            border: 1px solid #00ffe0 !important;
        }}

        .streamlit-expanderHeader {{
            color: #ffffff !important;
            background-color: #1a1a2a !important;
            border-radius: 5px !important;
            font-weight: 600 !important;
        }}

        .streamlit-expanderHeader:hover {{
            color: #00ffe0 !important;
        }}

        .streamlit-expanderContent {{
            background-color: #0f0f1a !important;
            border: 1px solid #00ffe0 !important;
            border-radius: 0 0 5px 5px !important;
        }}

        .streamlit-expanderContent * {{
            color: #ffffff !important;
        }}

        .stAlert {{
            background-color: #1a1a2a !important;
            border: 1px solid #00ffe0 !important;
            border-radius: 8px !important;
        }}

        .stAlert * {{
            color: #ffffff !important;
        }}

        .stInfo {{
            background-color: #1a3a4a !important;
            border-left-color: #00ffe0 !important;
        }}

        .stSuccess {{
            background-color: #1a3a2a !important;
            border-left-color: #00ff80 !important;
        }}

        .stWarning {{
            background-color: #4a3a1a !important;
            border-left-color: #ffaa00 !important;
        }}

        .stError {{
            background-color: #4a1a1a !important;
            border-left-color: #ff4444 !important;
        }}

        .stSpinner {{
            color: #00ffe0 !important;
        }}

        .stSpinner > div {{
            border-color: #00ffe0 transparent transparent transparent !important;
        }}

        .stButton button {{
            background: linear-gradient(45deg, #00a8a0, #00ffe0) !important;
            color: #000000 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 10px 25px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }}

        .stButton button:hover {{
            background: linear-gradient(45deg, #00ffe0, #ffd966) !important;
            transform: scale(1.05) !important;
            box-shadow: 0 0 20px rgba(0, 255, 224, 0.7) !important;
        }}

        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}

        ::-webkit-scrollbar-track {{
            background: #1a1a2a;
        }}

        ::-webkit-scrollbar-thumb {{
            background: #00ffe0;
            border-radius: 5px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: #ffd966;
        }}

        table {{
            color: #ffffff !important;
            background-color: #1a1a2a !important;
            border: 1px solid #00ffe0 !important;
        }}

        th {{
            background-color: #2a2a3a !important;
            color: #00ffe0 !important;
            font-weight: 700 !important;
            border-bottom: 2px solid #00ffe0 !important;
        }}

        td {{
            color: #ffffff !important;
            border-bottom: 1px solid #3a3a4a !important;
        }}

        tr:hover {{
            background-color: #2a2a3a !important;
        }}

        .stCaption, caption, .small, .st-caption {{
            color: #cccccc !important;
            font-style: italic !important;
        }}

        a {{
            color: #00ffe0 !important;
            text-decoration: none !important;
            font-weight: 600 !important;
        }}

        a:hover {{
            color: #ffd966 !important;
            text-decoration: underline !important;
        }}

        code {{
            background-color: #2a2a3a !important;
            color: #ffb86b !important;
            padding: 2px 5px !important;
            border-radius: 4px !important;
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
# FUNCIÓN PARA MOSTRAR RESPUESTA CON STREAMING
# ==========================

def mostrar_respuesta_streaming(mensajes):
    """Muestra la respuesta en tiempo real con efecto de escritura"""
    
    # Crear contenedor para la respuesta
    st.markdown("### 🤖 Respuesta de AguweyBot PRO")
    st.markdown("---")
    response_container = st.empty()
    
    # Crear el callback handler
    callback = StreamlitCallbackHandler(response_container)
    
    # Crear un nuevo LLM específicamente para streaming
    llm_stream = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        num_ctx=4096,
        top_p=0.9,
        repeat_penalty=1.1,
        streaming=True,
        callbacks=[callback]
    )
    
    # Generar respuesta con streaming
    response = llm_stream.invoke(mensajes)
    
    # Mostrar versión final sin cursor
    response_container.markdown(f'<div class="respuesta-aguwey">{response.content}</div>', unsafe_allow_html=True)
    
    return response


# ==========================
# INTERFAZ PRINCIPAL
# ==========================

st.set_page_config(
    page_title="AguweyBot PRO", 
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

set_background("fondo.png")

with st.sidebar:
    st.markdown("# ⚡ AGUWEYBOT PRO")
    st.markdown("### *Asistente Técnico Inteligente*")

    if os.path.exists("logo.png"):
        st.image("logo.png", width=220)
    else:
        st.info("📌 Coloca tu logo en `logo.png` para personalizar")

    st.markdown("---")
    st.markdown("### 🎯 Capacidades")
    st.markdown("""
    - 📚 RAG Semántico
    - 🔬 Análisis Técnico
    - 💡 Respuestas en Tiempo Real
    - 🎨 Visual Mejorado
    - ⚡ Streaming Avanzado
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Estado del Sistema")
    st.success("✅ Modelo: phi3:mini")
    st.info(f"📁 Vector DB: {PERSIST_DIRECTORY}")
    st.success("⚡ Streaming: Activado")
    
    st.markdown("---")
    st.caption("© 2024 AguweyBot PRO v3.0")
    st.caption("Arquitectura RAG Profesional con Streaming")

# Título principal
st.markdown("""
<style>
@keyframes glow {
    0% { text-shadow: 0 0 10px #00ffe0; }
    50% { text-shadow: 0 0 20px #00ffe0, 0 0 30px #ffd966; }
    100% { text-shadow: 0 0 10px #00ffe0; }
}

.titulo-principal {
    animation: glow 3s infinite;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="titulo-principal">⚡ AguweyBot PRO</h1>', unsafe_allow_html=True)
st.caption("Sistema cognitivo con recuperación semántica y generación en tiempo real")

# Cargar modelos
with st.spinner("🚀 Inicializando sistemas cognitivos..."):
    llm = cargar_llm()
    retriever = cargar_retriever()

# Input de pregunta
pregunta = st.text_input(
    "🔍 **Escribe tu consulta técnica:**",
    placeholder="Ej: ¿Cómo calcular la resistencia equivalente en un circuito paralelo?",
    key="pregunta_input"
)

# Procesar pregunta
if pregunta:
    # Mostrar spinner solo durante la recuperación de documentos
    with st.spinner("🧠 Buscando en la base de conocimiento..."):
        docs = retriever.invoke(pregunta)

    if not docs:
        # Sin contexto RAG
        mensajes = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=pregunta)
        ]
        
        # Usar streaming para la respuesta
        respuesta = mostrar_respuesta_streaming(mensajes)
        
        # Mostrar advertencia después de la respuesta
        st.info("ℹ️ No se encontraron documentos relevantes en la base de conocimiento. Respuesta basada en conocimiento general.")
        
    else:
        # Con contexto RAG
        contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        
        # Mostrar documentos recuperados en el sidebar
        with st.sidebar:
            st.markdown("### 📚 Documentos Recuperados")
            st.markdown(f"*Fuentes encontradas: {len(docs)}*")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"📄 Fuente {i}"):
                    st.markdown(f"**Contenido:**\n{doc.page_content[:200]}...")
        
        prompt_final = f"""
Contexto técnico relevante:
{contexto_rag}

Instrucciones:
- Analiza cuidadosamente el contexto proporcionado.
- Extrae únicamente la información pertinente a la pregunta.
- Responde con rigor técnico y estructura profesional.
- Si el contexto no contiene la respuesta completa, indícalo claramente.
- Fundamenta tu respuesta en los datos del contexto.

Pregunta del usuario:
{pregunta}
"""
        mensajes = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_final)
        ]
        
        # Usar streaming para la respuesta
        respuesta = mostrar_respuesta_streaming(mensajes)
        
        # Mostrar fuentes en la respuesta principal también
        with st.expander("📑 Ver fuentes completas"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Fuente {i}:**")
                st.markdown(f"```\n{doc.page_content}\n```")
                st.markdown("---")

else:
    # Mensaje de bienvenida
    st.info("👆 **¡Bienvenido!** Escribe tu pregunta técnica en el campo superior para comenzar.")
    
    # Ejemplos de preguntas
    with st.expander("💡 Ejemplos de preguntas que puedes hacer:"):
        st.markdown("""
        - "¿Cómo funciona un amplificador operacional en configuración no inversora?"
        - "Explica el teorema de superposición en circuitos eléctricos"
        - "¿Qué consideraciones debo tener para el diseño de un filtro pasa-bajos?"
        - "Calcula la potencia disipada en una resistencia de 100Ω con 5V"
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #cccccc;'>⚡ Desarrollado con tecnología RAG + LangChain + Ollama + Streaming</p>",
    unsafe_allow_html=True
)