# ============================================
# AGUWEYBOT PRO - RAG + VISUAL + STREAMING (VERSIÓN FINAL CORREGIDA)
# ============================================

import os
import base64
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- IMPORTACIÓN CORRECTA
import time

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_NAME = "phi3:mini"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "vector_db"
KNOWLEDGE_FILE = "conocimiento.txt"

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
"""

# ============================================
# CALLBACK PARA STREAMING
# ============================================
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

# ============================================
# FUNCIÓN PARA CARGAR CONOCIMIENTO.TXT
# ============================================
@st.cache_resource(show_spinner=False)
def crear_vectorstore():
    """
    Carga conocimiento.txt y crea el vector store si no existe
    """
    # Verificar si existe el archivo de conocimiento
    if not os.path.exists(KNOWLEDGE_FILE):
        st.warning(f"⚠️ No se encuentra el archivo {KNOWLEDGE_FILE}. Se usará solo el modelo base.")
        return None
    
    try:
        # Verificar si ya existe el vector store
        if os.path.exists(PERSIST_DIRECTORY):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
            return vectorstore
        
        # Si no existe, crearlo
        with st.spinner(f"📚 Procesando {KNOWLEDGE_FILE} por primera vez..."):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            
            # Cargar el documento
            loader = TextLoader(KNOWLEDGE_FILE, encoding='utf-8')
            documents = loader.load()
            
            # Dividir en chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Crear vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            vectorstore.persist()
            
            st.success(f"✅ Archivo {KNOWLEDGE_FILE} procesado: {len(chunks)} fragmentos")
            return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error al procesar {KNOWLEDGE_FILE}: {str(e)}")
        return None

# ============================================
# CARGA DE MODELOS
# ============================================
@st.cache_resource
def cargar_llm():
    try:
        return ChatOllama(
            model=MODEL_NAME,
            temperature=0.0,
            num_ctx=4096,
            top_p=0.9,
            repeat_penalty=1.1
        )
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo LLM: {str(e)}")
        return None

@st.cache_resource
def cargar_retriever():
    vectorstore = crear_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ============================================
# ESTILOS PROFESIONALES
# ============================================
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
            background-color: rgba(13, 17, 23, 0.85) !important;
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 2.5rem 1.5rem 6rem !important;
            max-width: 1100px !important;
            margin-top: 1rem;
            margin-bottom: 2rem;
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
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 1.5rem 2rem;
            margin: 1.5rem 0;
            line-height: 1.65;
            font-family: 'Segoe UI', system-ui, sans-serif;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        .respuesta-aguwey.streaming {{
            border-left-color: var(--accent);
        }}

        .respuesta-aguwey strong {{ color: var(--primary); }}
        
        .respuesta-aguwey code {{
            background: #0d1117;
            padding: 0.15em 0.4em;
            border-radius: 5px;
            font-family: 'Consolas', 'Courier New', monospace;
            color: var(--accent);
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

        h1 {{ 
            font-size: 2.4rem !important; 
            letter-spacing: -0.5px; 
            margin-bottom: 0.4rem !important;
            text-shadow: 0 0 20px rgba(0,212,255,0.3);
        }}
        
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
            font-size: 1rem;
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
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .stButton > button:hover {{
            background: linear-gradient(145deg, var(--primary), #4dd0ff);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0,212,255,0.25);
        }}

        .stAlert {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
        }}

        .cursor {{
            animation: blink 1s infinite;
            color: var(--primary);
            font-weight: bold;
        }}

        @keyframes blink {{
            0%, 50% {{ opacity: 1; }}
            51%, 100% {{ opacity: 0; }}
        }}

        .fixed-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(13,17,23,0.95);
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

        .fixed-footer a:hover {{ 
            text-decoration: underline; 
        }}

        .stExpander {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 0.5rem 0;
        }}

        .stExpander summary {{
            color: var(--primary);
            font-weight: 500;
        }}

        ::-webkit-scrollbar {{ 
            width: 8px; 
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{ 
            background: #0d1117; 
        }}
        
        ::-webkit-scrollbar-thumb {{ 
            background: #444d56; 
            border-radius: 4px; 
        }}
        
        ::-webkit-scrollbar-thumb:hover {{ 
            background: var(--primary-dark); 
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
        
        # Mostrar respuesta final sin cursor
        response_container.markdown(
            f'<div class="respuesta-aguwey">{response.content}</div>',
            unsafe_allow_html=True
        )
        
        return response
        
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
            st.caption("Crea este archivo para usar RAG")

        st.markdown("---")
        st.markdown("### 🎯 Capacidades")
        st.markdown("""
        - 📚 RAG con conocimiento.txt
        - 🔬 Análisis Técnico
        - ✍️ Escritura Creativa
        - 💡 Respuestas en Tiempo Real
        - 🎨 Visual Mejorado
        - ⚡ Streaming Avanzado
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Estado del Sistema")
        st.success(f"✅ Modelo: {MODEL_NAME}")
        st.info(f"📁 Vector DB: {PERSIST_DIRECTORY}")
        st.success("⚡ Streaming: Activado")
        
        st.markdown("---")
        st.caption("CC-NC-SA: 2026 AguweyBot PRO")
        st.caption("Arquitectura RAG Profesional")

    # Título principal
    st.markdown('<h1 style="color: #00d4ff;">⚡ AguweyBot PRO</h1>', unsafe_allow_html=True)
    st.caption("Sistema cognitivo con recuperación semántica y generación en tiempo real")

    # Cargar modelos
    with st.spinner("🚀 Inicializando sistemas cognitivos..."):
        llm = cargar_llm()
        retriever = cargar_retriever()

    if llm is None:
        st.error("❌ No se pudo cargar el modelo. Verifica que Ollama esté ejecutándose.")
        st.stop()

    # Entrada del usuario
    pregunta = st.text_input(
        "🔍 **Escribe tu consulta:**",
        placeholder="Ej: ¿Cómo calcular la resistencia equivalente en un circuito paralelo?",
        key="pregunta_input"
    )

    if pregunta:
        # Verificar si es saludo
        if pregunta.lower().strip() in ['hola', 'buenas', 'saludos', 'hey', 'hi', 'hello']:
            st.markdown("""
            <div class="respuesta-aguwey">
            👋 **¡Hola!** Soy AguweyBot PRO, tu asistente cognitivo.
            
            Puedo ayudarte con:
            - 🔬 Preguntas técnicas de ingeniería
            - ✍️ Desarrollo de escritura creativa
            - 📚 Consultas sobre tu base de conocimiento personalizada
            
            **¿En qué puedo ayudarte hoy?**
            </div>
            """, unsafe_allow_html=True)
            return

        with st.spinner("🧠 Buscando en la base de conocimiento..."):
            docs = retriever.invoke(pregunta) if retriever else []

        if not docs:
            mensajes = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=pregunta)
            ]
            mostrar_respuesta_streaming(mensajes)
            if retriever is None:
                st.info("ℹ️ Modo: Solo conocimiento general (sin documentos)")
            else:
                st.info("ℹ️ No se encontraron documentos relevantes. Usando conocimiento general.")
        else:
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
            
            with st.sidebar:
                st.markdown("### 📚 Documentos Recuperados")
                st.markdown(f"Fuentes encontradas: **{len(docs)}**")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Fuente {i}"):
                        st.markdown(doc.page_content[:350] + "..." if len(doc.page_content) > 350 else doc.page_content)

            prompt_final = f"""
Contexto relevante:
{contexto_rag}

Pregunta del usuario:
{pregunta}

Responde basándote en el contexto proporcionado. Si el contexto no contiene la información necesaria, indícalo claramente.
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
        st.info("👆 **¡Bienvenido!** Escribe tu pregunta arriba para comenzar.")
        
        with st.expander("💡 Ejemplos de preguntas", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔬 Técnicas:**")
                st.markdown("""
                - ¿Cómo funciona un amplificador operacional?
                - Explica el teorema de superposición
                - ¿Qué es la ley de Ohm?
                """)
            
            with col2:
                st.markdown("**✍️ Creativas:**")
                st.markdown("""
                - Ayúdame a desarrollar un personaje
                - ¿Cómo estructurar un cuento?
                - Consejos para diálogos realistas
                """)

    # Footer
    st.markdown("""
    <div class="fixed-footer">
        <strong>Licencia CC-NC-SA</strong> • Prof. Raymond Rosa Ávila • AguweyBot PRO 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()