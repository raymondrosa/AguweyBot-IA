# ============================================
# AGUWEYBOT PRO - CON MEMORIA Y CONTRASTE MEJORADO
# ============================================

import os
import base64
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_NAME = "phi3:mini"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "vector_db"
KNOWLEDGE_FILE = "conocimiento.txt"
MAX_HISTORY = 10

# ============================================
# SYSTEM PROMPT
# ============================================
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

IMPORTANTE: Tienes acceso al historial completo de la conversación. 
Úsalo para mantener coherencia con lo hablado anteriormente.
Recuerda detalles que el usuario te haya compartido antes.
"""

# ============================================
# CALLBACK PARA STREAMING
# ============================================
class StreamlitCallbackHandler(BaseCallbackHandler):
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

# ============================================
# FUNCIÓN PARA CARGAR CONOCIMIENTO.TXT
# ============================================
@st.cache_resource(show_spinner=False)
def crear_vectorstore():
    if not os.path.exists(KNOWLEDGE_FILE):
        st.warning(f"⚠️ No se encuentra el archivo {KNOWLEDGE_FILE}")
        return None
    
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
            return vectorstore
        
        with st.spinner(f"📚 Procesando {KNOWLEDGE_FILE} por primera vez..."):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            loader = TextLoader(KNOWLEDGE_FILE, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            vectorstore.persist()
            st.success(f"✅ Archivo procesado: {len(chunks)} fragmentos")
            return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ============================================
# CARGA DE MODELOS
# ============================================
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
    vectorstore = crear_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ============================================
# FUNCIÓN PARA CONSTRUIR MENSAJES CON HISTORIAL
# ============================================
def construir_mensajes_con_historial(pregunta, docs=None):
    mensajes = [SystemMessage(content=SYSTEM_PROMPT)]
    
    for msg in st.session_state.messages[-MAX_HISTORY:]:
        if msg["role"] == "user":
            mensajes.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            mensajes.append(AIMessage(content=msg["content"]))
    
    if docs:
        contexto = "\n\n".join([doc.page_content for doc in docs])
        mensajes.append(HumanMessage(content=f"Contexto:\n{contexto}\n\nPregunta actual: {pregunta}"))
    else:
        mensajes.append(HumanMessage(content=pregunta))
    
    return mensajes

# ============================================
# ESTILOS PROFESIONALES - CONTRASTE MEJORADO
# ============================================
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
            
        st.markdown(f"""
        <style>
        :root {{
            --primary: #00ffff;
            --primary-dark: #00cccc;
            --primary-bright: #7fffff;
            --accent: #ffaa00;
            --accent-bright: #ffcc66;
            --bg-dark: #0a0c10;
            --bg-card: #14181f;
            --bg-input: #1e242c;
            --text: #ffffff;
            --text-bright: #ffffff;
            --text-muted: #b0b8c5;
            --text-soft: #e0e5f0;
            --border: #3a4452;
            --border-light: #4e5a6b;
            --success: #2ea043;
            --danger: #f85149;
            --warning: #f0883e;
        }}

        html, body, .stApp {{
            background-color: var(--bg-dark);
            color: var(--text-bright);
        }}

        .main .block-container {{
            background-color: rgba(10, 12, 16, 0.92) !important;
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 2.5rem 1.5rem 6rem !important;
            max-width: 1100px !important;
            margin-top: 1rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}

        /* ===== RESPUESTAS DEL BOT - CONTRASTE MEJORADO ===== */
        .respuesta-aguwey {{
            background-color: #1a212b;
            border: 2px solid var(--border-light);
            border-left: 6px solid var(--primary);
            border-radius: 12px;
            padding: 1.8rem 2.2rem;
            margin: 1.8rem 0;
            line-height: 1.8;
            font-family: 'Segoe UI', system-ui, sans-serif;
            font-size: 1.1rem;
            color: var(--text-bright);
            box-shadow: 0 8px 24px rgba(0, 255, 255, 0.15);
        }}

        .respuesta-aguwey.streaming {{
            border-left-color: var(--accent);
            box-shadow: 0 8px 24px rgba(255, 170, 0, 0.15);
        }}

        .respuesta-aguwey p {{
            margin-bottom: 1rem;
            color: var(--text-bright);
        }}

        .respuesta-aguwey strong {{
            color: var(--primary-bright);
            font-weight: 700;
            text-shadow: 0 0 8px rgba(0, 255, 255, 0.5);
        }}
        
        .respuesta-aguwey em {{
            color: var(--accent-bright);
            font-style: italic;
        }}
        
        .respuesta-aguwey h1, .respuesta-aguwey h2, .respuesta-aguwey h3 {{
            color: var(--primary-bright) !important;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }}
        
        .respuesta-aguwey h1 {{ font-size: 1.8rem; }}
        .respuesta-aguwey h2 {{ font-size: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
        .respuesta-aguwey h3 {{ font-size: 1.3rem; }}
        
        .respuesta-aguwey code {{
            background: #0d1420;
            padding: 0.2em 0.5em;
            border-radius: 6px;
            font-family: 'Consolas', 'Courier New', monospace;
            color: var(--accent-bright);
            border: 1px solid var(--border);
            font-size: 0.95rem;
        }}

        .respuesta-aguwey pre {{
            background: #0d1420;
            border: 2px solid var(--border);
            padding: 1.5rem;
            border-radius: 12px;
            overflow-x: auto;
            color: var(--text-soft);
            font-size: 0.95rem;
            line-height: 1.6;
        }}
        
        .respuesta-aguwey pre code {{
            background: none;
            border: none;
            padding: 0;
            color: var(--text-soft);
        }}
        
        .respuesta-aguwey ul, .respuesta-aguwey ol {{
            margin: 1rem 0;
            padding-left: 2rem;
        }}
        
        .respuesta-aguwey li {{
            margin: 0.5rem 0;
            color: var(--text-soft);
        }}
        
        .respuesta-aguwey blockquote {{
            border-left: 4px solid var(--primary);
            background: rgba(0, 255, 255, 0.1);
            padding: 0.8rem 1.5rem;
            margin: 1.2rem 0;
            border-radius: 0 12px 12px 0;
            color: var(--text-soft);
            font-style: italic;
        }}
        
        .respuesta-aguwey a {{
            color: var(--primary-bright);
            text-decoration: underline;
            text-decoration-color: var(--primary);
        }}
        
        .respuesta-aguwey a:hover {{
            color: var(--accent-bright);
        }}
        
        .respuesta-aguwey table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1.2rem 0;
            color: var(--text-soft);
        }}
        
        .respuesta-aguwey th {{
            background: var(--primary-dark);
            color: var(--bg-dark);
            font-weight: 700;
            padding: 0.8rem;
            border: 1px solid var(--border);
        }}
        
        .respuesta-aguwey td {{
            padding: 0.6rem 0.8rem;
            border: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.2);
        }}

        /* ===== MENSAJES DEL USUARIO ===== */
        .user-message {{
            background: linear-gradient(145deg, #1e2a3a, #15232e);
            border: 2px solid var(--primary);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            color: var(--text-bright);
            font-size: 1.1rem;
        }}
        
        .user-message strong {{
            color: var(--primary-bright);
        }}

        /* ===== TÍTULOS ===== */
        h1, h2, h3 {{
            color: var(--primary-bright) !important;
            font-weight: 700;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
        }}

        h1 {{ 
            font-size: 2.6rem !important; 
            letter-spacing: -0.5px; 
            margin-bottom: 0.4rem !important;
        }}
        
        h2 {{
            font-size: 2rem !important;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem;
        }}

        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background: linear-gradient(165deg, #0e1219, #0a0e14) !important;
            border-right: 2px solid var(--primary);
        }}

        [data-testid="stSidebar"] .stMarkdown {{
            color: var(--text-soft);
        }}

        [data-testid="stSidebar"] h1 {{
            color: var(--primary-bright) !important;
            font-size: 2rem !important;
            text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        }}

        /* ===== INPUT DE TEXTO ===== */
        .stTextInput input {{
            background-color: var(--bg-input);
            color: var(--text-bright);
            border: 2px solid var(--border);
            border-radius: 12px;
            padding: 0.9rem 1.2rem;
            font-size: 1.1rem;
            font-weight: 500;
        }}

        .stTextInput input:focus {{
            border-color: var(--primary);
            border-width: 2px;
            box-shadow: 0 0 0 4px rgba(0, 255, 255, 0.25);
        }}
        
        .stTextInput input::placeholder {{
            color: var(--text-muted);
            font-weight: 400;
        }}

        /* ===== BOTONES ===== */
        .stButton > button {{
            background: linear-gradient(145deg, var(--primary-dark), var(--primary));
            color: var(--bg-dark) !important;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            transition: all 0.2s ease;
            border: 1px solid var(--primary-bright);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stButton > button:hover {{
            background: linear-gradient(145deg, var(--primary), var(--primary-bright));
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 255, 255, 0.4);
        }}

        /* ===== CHAT MESSAGES ===== */
        .chat-message-user {{
            background: linear-gradient(145deg, #1f2b38, #17222b);
            border: 2px solid var(--primary);
            border-radius: 15px 15px 4px 15px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            color: var(--text-bright);
        }}
        
        .chat-message-bot {{
            background: linear-gradient(145deg, #1a212b, #131a22);
            border: 2px solid var(--accent);
            border-radius: 15px 15px 15px 4px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            color: var(--text-bright);
        }}

        /* ===== EXPANDER ===== */
        .stExpander {{
            background-color: var(--bg-card);
            border: 2px solid var(--border);
            border-radius: 12px;
            margin: 1rem 0;
        }}

        .stExpander summary {{
            color: var(--primary-bright);
            font-weight: 600;
            font-size: 1.1rem;
            padding: 0.5rem;
        }}
        
        .stExpander .stMarkdown {{
            color: var(--text-soft);
        }}

        /* ===== CURSOR ===== */
        .cursor {{
            animation: blink 1s infinite;
            color: var(--primary-bright);
            font-weight: bold;
            font-size: 1.2rem;
        }}

        @keyframes blink {{
            0%, 50% {{ opacity: 1; }}
            51%, 100% {{ opacity: 0; }}
        }}

        /* ===== FOOTER ===== */
        .fixed-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(10, 12, 16, 0.98);
            backdrop-filter: blur(12px);
            border-top: 2px solid var(--primary);
            padding: 1rem 2rem;
            text-align: center;
            color: var(--text-soft);
            font-size: 0.95rem;
            font-weight: 500;
            z-index: 999;
        }}

        .fixed-footer strong {{
            color: var(--primary-bright);
        }}

        .fixed-footer a {{
            color: var(--primary-bright);
            text-decoration: none;
            font-weight: 600;
        }}

        .fixed-footer a:hover {{ 
            text-decoration: underline;
            color: var(--accent-bright);
        }}

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{ 
            width: 10px; 
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{ 
            background: var(--bg-dark); 
        }}
        
        ::-webkit-scrollbar-thumb {{ 
            background: linear-gradient(145deg, var(--primary-dark), var(--primary));
            border-radius: 5px; 
        }}
        
        ::-webkit-scrollbar-thumb:hover {{ 
            background: linear-gradient(145deg, var(--primary), var(--primary-bright));
        }}

        /* ===== INFO BOXES ===== */
        .stAlert {{
            background-color: var(--bg-card);
            border: 2px solid var(--border);
            border-radius: 10px;
            color: var(--text-soft);
        }}
        
        .stInfo {{
            background-color: rgba(0, 255, 255, 0.15);
            border-left-color: var(--primary);
            color: var(--text-bright);
        }}
        
        .stSuccess {{
            background-color: rgba(46, 160, 67, 0.15);
            border-left-color: var(--success);
            color: var(--text-bright);
        }}
        
        .stWarning {{
            background-color: rgba(240, 136, 62, 0.15);
            border-left-color: var(--warning);
            color: var(--text-bright);
        }}
        
        .stError {{
            background-color: rgba(248, 81, 73, 0.15);
            border-left-color: var(--danger);
            color: var(--text-bright);
        }}

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1.5rem 1rem 6rem !important;
            }}
            
            h1 {{
                font-size: 2.2rem !important;
            }}
            
            .respuesta-aguwey {{
                padding: 1.2rem 1.5rem;
                font-size: 1rem;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)

# ============================================
# STREAMING DE RESPUESTA
# ============================================
def mostrar_respuesta_streaming(mensajes):
    try:
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
        
        return response.content
        
    except Exception as e:
        st.error(f"❌ Error en streaming: {str(e)}")
        return None

# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def main():
    st.set_page_config(
        page_title="AguweyBot PRO",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_background("fondo.png")

    # Inicializar session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "first_interaction" not in st.session_state:
        st.session_state.first_interaction = True

    # Sidebar
    with st.sidebar:
        st.markdown("# ⚡AGUWEYBOT")
        st.markdown("### *Asistente Inteligente*")

        if os.path.exists("logo.png"):
            st.image("logo.png", width=220)
        else:
            st.info("📌 Coloca 'logo.png' para personalizar")

        st.markdown("---")
        
        # Información del conocimiento
        st.markdown("### 📚 Conocimiento Base")
        if os.path.exists(KNOWLEDGE_FILE):
            file_size = os.path.getsize(KNOWLEDGE_FILE) / 1024
            st.success(f"✅ {KNOWLEDGE_FILE} ({file_size:.1f} KB)")
        else:
            st.warning(f"⚠️ {KNOWLEDGE_FILE} no encontrado")

        st.markdown("---")
        
        # Estadísticas
        st.markdown("### 💬 Conversación")
        st.markdown(f"**Mensajes:** {len(st.session_state.messages)}")
        
        if len(st.session_state.messages) > 0:
            if st.button("🔄 Nueva Conversación"):
                st.session_state.messages = []
                st.session_state.first_interaction = True
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 Capacidades")
        st.markdown("""
        - 📚 RAG con conocimiento.txt
        - 🔬 Análisis Técnico
        - ✍️ Escritura Creativa
        - 💬 Memoria de Conversación
        - ⚡ Streaming Avanzado
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Estado del Sistema")
        st.success(f"✅ Modelo: {MODEL_NAME}")
        st.info(f"📁 Memoria: {len(st.session_state.messages)} mensajes")
        st.success("⚡ Streaming: Activado")
        
        st.markdown("---")
        st.caption("CC-NC-SA: 2026 AguweyBot PRO")

    # Contenido principal
    st.markdown('<h1 style="color: #00ffff; text-shadow: 0 0 20px rgba(0,255,255,0.5);">⚡ AguweyBot PRO</h1>', unsafe_allow_html=True)
    st.caption("Sistema cognitivo con memoria de conversación y recuperación semántica")

    # Cargar modelos
    with st.spinner("🚀 Inicializando sistemas cognitivos..."):
        llm = cargar_llm()
        retriever = cargar_retriever()

    if llm is None:
        st.error("❌ No se pudo cargar el modelo. Verifica Ollama.")
        st.stop()

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    prompt = st.chat_input("Escribe tu consulta...")

    if prompt:
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Guardar en historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Buscar documentos
        with st.spinner("🧠 Pensando..."):
            docs = retriever.invoke(prompt) if retriever else []
        
        # Construir mensajes
        mensajes = construir_mensajes_con_historial(prompt, docs if docs else None)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            respuesta = mostrar_respuesta_streaming(mensajes)
        
        # Guardar respuesta
        if respuesta:
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
        
        # Mostrar fuentes
        if docs:
            with st.expander("📚 Fuentes consultadas"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Fuente {i}:**")
                    st.caption(doc.page_content[:200] + "...")

    # Mensaje de bienvenida
    elif st.session_state.first_interaction:
        st.info("👋 **¡Bienvenido!** Puedes hacerme preguntas técnicas o literarias. Recuerdo toda la conversación, así que podemos profundizar en temas complejos.")
        st.session_state.first_interaction = False

    # Footer
    st.markdown("""
    <div class="fixed-footer">
        <strong>⚡ Licencia CC-NC-SA</strong> • Prof. Raymond Rosa Ávila • AguweyBot PRO 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()