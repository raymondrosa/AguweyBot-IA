# ============================================
# AGUWEYBOT PRO - RAG + VISUAL + STREAMING (ESTILO PROFESIONAL MEJORADO)
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

Eres un asistente avanzado con doble especialización:

1. Ingeniería y ciencias aplicadas.
2. Escritura creativa, narrativa y desarrollo literario.

Tu comportamiento depende del tipo de consulta del usuario.

MODO TÉCNICO:
- Usa rigor científico.
- Fundamenta en principios verificables.
- Estructura en secciones.
- Mantén precisión conceptual.
- No inventes datos.

MODO CREATIVO:
- Puedes asistir en escritura literaria.
- Ayuda con desarrollo de personajes.
- Construcción de tramas.
- Mejora de estilo.
- Redacción narrativa.
- Estructura de capítulos.
- Diálogos.
- Correcciones literarias.

REGLAS GENERALES:
- Detecta automáticamente si la consulta es técnica o creativa.
- Si es técnica → responde con rigor ingenieril.
- Si es literaria → responde con enfoque narrativo profesional.
- Nunca rechaces ayudar en escritura creativa.
- No menciones limitaciones innecesarias.
- Mantén coherencia y calidad en ambos modos.
- Si falta información, pide aclaración de forma profesional.

Tu objetivo es ser un asistente cognitivo integral de alto nivel.

DIRECTRICES DE ESTILO:
- Utiliza emojis estratégicamente para mejorar la comunicación visual
- En modo técnico: usa emojis para secciones (🔬, 📊, ⚙️, 📐)
- En modo creativo: usa emojis expresivos (✍️, 📖, 🎭, ✨)
- No sobrecargues el texto con emojis innecesarios
- Mantén un balance profesional entre texto y elementos visuales
- Prefiere emojis al inicio de secciones o para destacar puntos clave

EJEMPLOS DE USO APROPIADO:
🔍 Análisis Técnico:
📌 Consideraciones importantes:
💡 Recomendación:
⚠️ Precaución:
✅ Verificación:
"""
# ────────────────────────────────────────────
# CALLBACK PARA STREAMING EN STREAMLIT
# ────────────────────────────────────────────

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler para streaming de tokens en Streamlit"""
    
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(
            f'<div class="respuesta-aguwey streaming">{self.text}<span class="cursor">▌</span></div>',
            unsafe_allow_html=True
        )
        time.sleep(0.005)

# ────────────────────────────────────────────
# ESTILOS PROFESIONALES - TEMA TÉCNICO OSCURO
# ────────────────────────────────────────────

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(f"""
        <style>
        :root {{
            --primary: #00d4ff;
            --primary-dark: #00a0c7;
            --accent: #ffd54f;
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --border: #30363d;
        }}

        html, body, .stApp {{
            background-color: var(--bg-dark);
            color: var(--text);
        }}

        .main .block-container {{
            background-color: transparent !important;
            padding: 2.5rem 1.5rem 6rem !important;
            max-width: 1100px !important;
        }}

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        .respuesta-aguwey {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin: 1.5rem 0;
            line-height: 1.65;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }}

        .respuesta-aguwey strong {{ color: var(--primary); }}
        .respuesta-aguwey code {{
            background: #0d1117;
            padding: 0.15em 0.4em;
            border-radius: 5px;
            font-family: 'Consolas', 'Courier New', monospace;
        }}

        .respuesta-aguwey pre {{
            background: #0d1117;
            border: 1px solid var(--border);
            padding: 1.2rem;
            border-radius: 8px;
            overflow-x: auto;
        }}

        h1, h2, h3 {{
            color: var(--primary) !important;
            font-weight: 600;
        }}

        h1 {{ font-size: 2.4rem !important; letter-spacing: -0.5px; margin-bottom: 0.4rem !important; }}
        h2 {{
            font-size: 1.8rem !important;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem;
        }}

        [data-testid="stSidebar"] {{
            background-color: #0a0e14 !important;
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] h1 {{
            color: var(--primary) !important;
            font-size: 1.9rem !important;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 0.2rem;
        }}

        [data-testid="stSidebar"] h3 {{
            color: var(--text-muted);
            font-size: 0.95rem;
            font-weight: 400;
            margin-top: 0;
        }}

        .stTextInput input {{
            background-color: #0d1117;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }}

        .stTextInput input:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(0,212,255,0.15);
        }}

        .stButton > button {{
            background: linear-gradient(145deg, var(--primary-dark), var(--primary));
            color: black !important;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.65rem 1.4rem;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            background: linear-gradient(145deg, var(--primary), #4dd0ff);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0,212,255,0.25);
        }}

        button[title="Rerun"],
        button[kind="secondary"] {{
            background-color: #238636 !important;
            color: white !important;
            border: none !important;
        }}

        button[title="Stop"],
        button[kind="primary"] {{
            background-color: #da3633 !important;
            color: white !important;
            border: none !important;
        }}

        .fixed-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(13,17,23,0.92);
            backdrop-filter: blur(8px);
            border-top: 1px solid var(--border);
            padding: 0.9rem 2rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            z-index: 999;
        }}

        .fixed-footer a {{
            color: var(--primary);
            text-decoration: none;
        }}

        .fixed-footer a:hover {{ text-decoration: underline; }}

        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: #0d1117; }}
        ::-webkit-scrollbar-thumb {{ background: #444d56; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--primary-dark); }}
        </style>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────
# CARGA DE MODELOS (cacheados)
# ────────────────────────────────────────────

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

# ────────────────────────────────────────────
# STREAMING DE RESPUESTA
# ────────────────────────────────────────────

def mostrar_respuesta_streaming(mensajes):
    response_container = st.empty()
    callback = StreamlitCallbackHandler(response_container)
    
    llm_stream = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        num_ctx=4096,
        top_p=0.9,
        repeat_penalty=1.1,
        streaming=True,
        callbacks=[callback]
    )
    
    response = llm_stream.invoke(mensajes)
    response_container.markdown(
        f'<div class="respuesta-aguwey">{response.content}</div>',
        unsafe_allow_html=True
    )
    return response

# ────────────────────────────────────────────
# INTERFAZ PRINCIPAL
# ────────────────────────────────────────────

st.set_page_config(
    page_title="AguweyBot PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

set_background("fondo.png")

with st.sidebar:
    st.markdown("# ⚡AGUWEYBOT")
    st.markdown("### *Asistente Inteligente*")

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
    st.caption("CC-NC-SA: 2026 AguweyBot PRO")
    st.caption("Arquitectura RAG Profesional con Streaming")

# Título principal
st.markdown('<h1 style="color: #00d4ff; letter-spacing: -0.5px;">⚡ AguweyBot PRO</h1>', unsafe_allow_html=True)
st.caption("Sistema cognitivo con recuperación semántica y generación en tiempo real")

# Cargar modelos
with st.spinner("🚀 Inicializando sistemas cognitivos..."):
    llm = cargar_llm()
    retriever = cargar_retriever()

# Entrada del usuario
pregunta = st.text_input(
    "🔍 **Escribe tu consulta técnica:**",
    placeholder="Ej: ¿Cómo calcular la resistencia equivalente en un circuito paralelo?",
    key="pregunta_input"
)

if pregunta:
    with st.spinner("🧠 Buscando en la base de conocimiento..."):
        docs = retriever.invoke(pregunta)

    if not docs:
        mensajes = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=pregunta)
        ]
        mostrar_respuesta_streaming(mensajes)
        st.info("ℹ️ No se encontraron documentos relevantes. Respuesta basada en conocimiento general del modelo.")
    else:
        contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        
        with st.sidebar:
            st.markdown("### 📚 Documentos Recuperados")
            st.markdown(f"Fuentes encontradas: **{len(docs)}**")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Fuente {i}"):
                    st.markdown(doc.page_content[:350] + "..." if len(doc.page_content) > 350 else doc.page_content)

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
        
        mostrar_respuesta_streaming(mensajes)
        
        with st.expander("📑 Ver fuentes completas"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Fuente {i}:**")
                st.code(doc.page_content, language=None)
                st.markdown("---")
else:
    st.info("👆 **¡Bienvenido!** Escribe tu pregunta técnica arriba para comenzar.")
    
    with st.expander("💡 Ejemplos de preguntas que puedes hacer:"):
        st.markdown("""
        - ¿Cómo funciona un amplificador operacional en configuración no inversora?
        - Explica el teorema de superposición en circuitos eléctricos
        - ¿Qué consideraciones debo tener para el diseño de un filtro pasa-bajos?
        - Calcula la potencia disipada en una resistencia de 100 Ω con 5 V
        """)

# Footer con licencia
st.markdown("""
<div class="fixed-footer">
    <strong>Licencia:</strong> CC-NC-SA  
      •  
    Prof. Raymond Rosa Ávila  
      •  
    AguweyBot PRO 2026
</div>
""", unsafe_allow_html=True)