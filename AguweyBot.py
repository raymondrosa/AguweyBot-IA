# ============================================
#  AGUWEYBOT - ULTRA ECO PRO (4GB + STREAM)
# Profesional | Visual | Streaming
# ============================================

import os
import base64
import logging
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==========================
# CONFIGURACIÓN
# ==========================

MODEL_NAME = "phi3:mini"
CHAT_MEMORY_FILE = "chat_ultraeco_pro.txt"
MAX_HISTORY = 10  # Aumentado para más contexto y estabilidad

SYSTEM_PROMPT = """Eres AguweyBot, un asistente profesional, técnico y preciso.
Respondes en español claro y bien estructurado.
Tu enfoque es lógico y organizado.
Si no sabes algo, lo dices claramente.
No inventas información. Base tus respuestas solo en el prompt proporcionado, la historia de la conversación y conocimiento factual verificado.
No agregues o asumas hechos extras para evitar errores.
Cuida la ortografía y redacción.
Piensa paso a paso antes de responder para asegurar coherencia."""

# Configura logging para depurar erraticidad
logging.basicConfig(level=logging.INFO, filename='AguweyBot_log.txt', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================
# ESTILO Y FONDO
# ==========================

def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

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
            color: #f5f5f5 !important;
        }}

        .stTextInput input {{
            background-color: white !important;
            color: black !important;
            border-radius: 10px;
            border: 2px solid #00ffe0 !important;
        }}

        .user-bubble {{
            background-color: #004d40;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
            color: white;
        }}

        .ai-bubble {{
            background-color: #1a1a1a;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #00ffe0;
            color: white;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.9);
        }}

        /* ===== LICENCIA FIJA ===== */

        .licencia-footer {{
            position: fixed;
            bottom: 10px;
            right: 20px;
            font-size: 13px;
            color: rgba(0,255,224,0.6);
            font-weight: bold;
            z-index: 9999;
            pointer-events: none;
        }}

        </style>

        <div class="licencia-footer">
            Licencia: 2602264703505 | CC-NC | Prof. Raymond Rosa Ávila
        </div>

        """, unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Error cargando fondo: {e}")
        st.warning("No se pudo cargar el fondo.")

# ==========================
# MEMORIA SIMPLE
# ==========================

def cargar_memoria():
    mensajes = []
    if os.path.exists(CHAT_MEMORY_FILE):
        try:
            with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("Usuario:"):
                    mensajes.append(HumanMessage(content=line.replace("Usuario:", "").strip()))
                elif line.startswith("Asistente:"):
                    mensajes.append(AIMessage(content=line.replace("Asistente:", "").strip()))
        except Exception as e:
            logging.error(f"Error cargando memoria: {e}")
    return mensajes

def guardar_memoria(pregunta, respuesta):
    try:
        with open(CHAT_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"Usuario: {pregunta}\n")
            f.write(f"Asistente: {respuesta}\n")
    except Exception as e:
        logging.error(f"Error guardando memoria: {e}")

# ==========================
# STREAMLIT CONFIG
# ==========================

st.set_page_config(
    page_title=" AGUWEYBOT - Ultra Eco PRO",
    page_icon="⚡",
    layout="centered"
)

set_background("fondo.png")

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.markdown("## 🤖  AGUWEYBOT")

    if os.path.exists("logo.png"):
        st.image("logo.png", width=250)
    else:
        st.info("Coloca logo.png en la carpeta")

    st.markdown("---")
    st.markdown("Ultra Eco PRO + Streaming ⚡")

# ==========================
# CARGAR MODELO
# ==========================

@st.cache_resource
def cargar_modelo():
    try:
        return ChatOllama(
            model=MODEL_NAME,
            temperature=0.0,
            num_ctx=4096,  # Aumentado para más estabilidad
            top_p=0.95,    # Controla diversidad sin erraticidad
            repeat_penalty=1.1  # Evita repeticiones y divagaciones
        )
    except Exception as e:
        logging.error(f"Error cargando modelo: {e}")
        raise

try:
    llm = cargar_modelo()
except Exception:
    st.error("Modelo no encontrado. Ejecuta: ollama pull phi3:mini")
    st.stop()

# ==========================
# SESIÓN
# ==========================

if "mensajes" not in st.session_state:
    st.session_state.mensajes = cargar_memoria()

# ==========================
# INTERFAZ PRINCIPAL
# ==========================

st.title("⚡  AGUWEYBOT - Ultra Eco PRO")
st.caption("Arquitectura Profesional Optimizada")

pregunta = st.text_input("Escribe tu pregunta:")

if pregunta:
    with st.spinner("Analizando..."):
        try:
            historial = st.session_state.mensajes[-MAX_HISTORY:]

            contexto = (
                [SystemMessage(content=SYSTEM_PROMPT)]
                + historial
                + [HumanMessage(content=pregunta)]
            )

            logging.info(f"Contexto enviado: {contexto}")  # Para depurar

            placeholder = st.empty()
            respuesta_completa = ""

            for chunk in llm.stream(contexto):
                if hasattr(chunk, "content") and chunk.content:
                    respuesta_completa += chunk.content
                    placeholder.markdown(
                        f"<div class='ai-bubble'>{respuesta_completa}▌</div>",
                        unsafe_allow_html=True
                    )

            placeholder.markdown(
                f"<div class='ai-bubble'>{respuesta_completa}</div>",
                unsafe_allow_html=True
            )

            st.session_state.mensajes.append(HumanMessage(content=pregunta))
            st.session_state.mensajes.append(AIMessage(content=respuesta_completa))

            guardar_memoria(pregunta, respuesta_completa)

        except Exception as e:
            logging.error(f"Error de ejecución: {e}")
            st.error(f"Error de ejecución: {e}")

# ==========================
# HISTORIAL
# ==========================

with st.expander("Historial reciente"):
    for msg in st.session_state.mensajes[-MAX_HISTORY:]:
        if isinstance(msg, HumanMessage):
            st.markdown(
                f"<div class='user-bubble'><strong>Usuario:</strong><br>{msg.content}</div>",
                unsafe_allow_html=True
            )
        elif isinstance(msg, AIMessage):
            st.markdown(
                f"<div class='ai-bubble'><strong>AguweyBot:</strong><br>{msg.content}</div>",
                unsafe_allow_html=True
            )

st.markdown("---")
st.markdown(
    "<span style='color:#00ffe0;font-weight:bold;'>Sistema Ultra Eco PRO Activo</span>",
    unsafe_allow_html=True
)