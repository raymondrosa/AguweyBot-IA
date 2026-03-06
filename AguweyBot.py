# ============================================
# AGUWEYBOT PRO - CON MEMORIA, RAG Y ANÁLISIS DE DOCUMENTOS
# VERSIÓN MEJORADA CON ANÁLISIS NUMÉRICO
# ============================================

import os
import base64
import time
import streamlit as st
import re
import io

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NUEVOS IMPORTS PARA DOCUMENTOS
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract

# NUEVOS IMPORTS PARA MEJORAS NUMÉRICAS
import chardet
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_NAME = "phi3:mini"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "vector_db"
KNOWLEDGE_FILE = "conocimiento.txt"
MAX_HISTORY = 10

# ============================================
# SYSTEM PROMPT (mejorado para datos numéricos)
# ============================================
SYSTEM_PROMPT = """
Eres AguweyBot PRO.
Eres un asistente avanzado con doble especialización:
1. Ingeniería y ciencias aplicadas.
2. Escritura creativa, narrativa y desarrollo literario.

ADEMÁS, ERES EXPERTO EN ANÁLISIS DE DATOS NUMÉRICOS:
- Puedes interpretar tablas, estadísticas y series de datos
- Identificas patrones, tendencias y anomalías en datos numéricos
- Explicas conceptos estadísticos de forma clara
- Ayudas a visualizar mentalmente distribuciones de datos
- Puedes hacer cálculos aproximados y estimaciones

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

MODO ANÁLISIS DE DATOS:
- Identifica patrones numéricos y tendencias
- Calcula estadísticas descriptivas (media, mediana, moda, desviación)
- Detecta correlaciones y outliers
- Sugiere visualizaciones apropiadas
- Interpreta resultados en contexto

REGLAS GENERALES:
- Detecta automáticamente si la consulta es técnica, creativa o de análisis de datos.
- Si contiene tablas o números → activa modo análisis de datos.
- Nunca rechaces ayudar en escritura creativa.
- No menciones limitaciones innecesarias.
- Mantén coherencia y calidad en todos los modos.
- Si falta información, pide aclaración de forma profesional.

DIRECTRICES DE ESTILO:
- Utiliza emojis estratégicamente para mejorar la comunicación visual
- En modo técnico: usa emojis para secciones (🔬, 📊, ⚙️, 📐)
- En modo creativo: usa emojis expresivos (✍️, 📖, 🎭, ✨)
- En modo análisis de datos: usa (📈, 📉, 📊, 🔢, 📋)
- No sobrecargues el texto con emojis innecesarios
- Mantén un balance profesional entre texto y elementos visuales

EJEMPLOS DE USO APROPIADO:
🔍 Análisis Técnico:
📌 Consideraciones importantes:
💡 Recomendación:
⚠️ Precaución:
✅ Verificación:
📊 Análisis de Datos:
📈 Tendencia observada:
🔢 Estadísticas clave:

IMPORTANTE: Tienes acceso al historial completo de la conversación.
Úsalo para mantener coherencia con lo hablado anteriormente.
Recuerda detalles que el usuario te haya compartido antes.
"""

# ============================================
# CLASE PARA DATOS NUMÉRICOS ESTRUCTURADOS
# ============================================
class DatosNumericos:
    """Clase para manejar datos numéricos extraídos de documentos"""
    
    def __init__(self):
        self.dataframes = {}
        self.estadisticas = {}
        self.numeros_extraidos = []
        self.fuente = ""
        self.tipo_archivo = ""
    
    def agregar_dataframe(self, nombre, df):
        """Agrega un DataFrame y calcula sus estadísticas"""
        self.dataframes[nombre] = df
        self._calcular_estadisticas(nombre, df)
    
    def _calcular_estadisticas(self, nombre, df):
        """Calcula estadísticas para columnas numéricas"""
        stats = {}
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        for col in columnas_numericas:
            stats[col] = {
                "tipo": "numerica",
                "media": float(df[col].mean()) if not df[col].isna().all() else None,
                "mediana": float(df[col].median()) if not df[col].isna().all() else None,
                "moda": df[col].mode().tolist() if not df[col].mode().empty else [],
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
                "desviacion": float(df[col].std()) if not df[col].isna().all() else None,
                "varianza": float(df[col].var()) if not df[col].isna().all() else None,
                "cuartiles": df[col].quantile([0.25, 0.5, 0.75]).tolist() if not df[col].isna().all() else [],
                "valores_nulos": int(df[col].isna().sum()),
                "valores_unicos": int(df[col].nunique())
            }
        
        # Detectar posibles outliers
        for col in columnas_numericas:
            if not df[col].isna().all():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
                stats[col]["outliers"] = len(outliers)
                stats[col]["porcentaje_outliers"] = float(len(outliers) / len(df) * 100) if len(df) > 0 else 0
        
        self.estadisticas[nombre] = stats
    
    def extraer_numeros_texto(self, texto):
        """Extrae todos los números de un texto"""
        numeros = re.findall(r'-?\d+\.?\d*', texto)
        self.numeros_extraidos = [float(n) for n in numeros]
        return self.numeros_extraidos
    
    def generar_resumen_estadistico(self):
        """Genera un resumen de todas las estadísticas"""
        resumen = []
        
        if self.dataframes:
            resumen.append("📊 **DATOS ESTRUCTURADOS ENCONTRADOS**")
            for nombre_df, df in self.dataframes.items():
                resumen.append(f"\n--- {nombre_df} ---")
                resumen.append(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
                
                if nombre_df in self.estadisticas:
                    for col, stats in self.estadisticas[nombre_df].items():
                        resumen.append(f"\n  📈 {col}:")
                        resumen.append(f"    Media: {stats['media']:.2f}" if stats['media'] else "    Media: N/A")
                        resumen.append(f"    Mediana: {stats['mediana']:.2f}" if stats['mediana'] else "    Mediana: N/A")
                        resumen.append(f"    Min-Max: {stats['min']} - {stats['max']}" if stats['min'] else "    Min-Max: N/A")
                        resumen.append(f"    Desviación: {stats['desviacion']:.2f}" if stats['desviacion'] else "    Desviación: N/A")
        
        if self.numeros_extraidos:
            resumen.append("\n🔢 **NÚMEROS EXTRAÍDOS DE TEXTO**")
            resumen.append(f"Total: {len(self.numeros_extraidos)} valores")
            if self.numeros_extraidos:
                resumen.append(f"Rango: {min(self.numeros_extraidos):.2f} - {max(self.numeros_extraidos):.2f}")
                resumen.append(f"Media: {np.mean(self.numeros_extraidos):.2f}")
                resumen.append(f"Mediana: {np.median(self.numeros_extraidos):.2f}")
        
        return "\n".join(resumen)
    
    def generar_visualizacion(self, tipo="histograma", columna=None, nombre_df=None):
        """Genera una visualización de los datos"""
        try:
            if nombre_df and nombre_df in self.dataframes:
                df = self.dataframes[nombre_df]
                
                if tipo == "histograma" and columna and columna in df.columns:
                    fig = px.histogram(df, x=columna, title=f"Histograma de {columna}")
                    return fig
                
                elif tipo == "boxplot" and columna and columna in df.columns:
                    fig = px.box(df, y=columna, title=f"Boxplot de {columna}")
                    return fig
                
                elif tipo == "correlacion":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        fig = px.imshow(numeric_df.corr(), 
                                       text_auto=True, 
                                       title="Matriz de Correlación",
                                       color_continuous_scale="RdBu")
                        return fig
                
                elif tipo == "lineas" and len(df.columns) >= 2:
                    # Asumir primera columna como índice para gráfico de líneas
                    fig = px.line(df, x=df.columns[0], y=df.columns[1:].tolist(), 
                                 title="Gráfico de Líneas")
                    return fig
            
            return None
        except Exception as e:
            st.error(f"Error generando visualización: {e}")
            return None

# ============================================
# CALLBACK PARA STREAMING
# ============================================
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback para ir mostrando los tokens en tiempo real en Streamlit."""
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Se ejecuta cada vez que llega un nuevo token del modelo."""
        self.text += token
        self.container.markdown(
            f'<div class="respuesta-aguwey streaming">{self.text}<span class="cursor">▌</span></div>',
            unsafe_allow_html=True,
        )
        # Ajusta este delay si quieres que la escritura sea más rápida o más lenta.
        time.sleep(0.005)

# ============================================
# FUNCIÓN PARA CARGAR / CREAR VECTORSTORE (RAG) BASE
# ============================================
@st.cache_resource(show_spinner=False)
def crear_vectorstore():
    """Crea o carga la base vectorial a partir de conocimiento.txt."""
    if not os.path.exists(KNOWLEDGE_FILE):
        st.warning(f"⚠️ No se encuentra el archivo {KNOWLEDGE_FILE}")
        return None

    try:
        # Si ya existe la carpeta de persistencia, solo cargamos
        if os.path.exists(PERSIST_DIRECTORY):
            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
            )
            return vectorstore
        else:
            # Primera vez: procesar conocimiento.txt y crear la base de vectores
            progress_bar = st.progress(0, text="📚 Iniciando procesamiento de conocimiento.txt...")
            
            with st.spinner("📚 Procesando conocimiento.txt por primera vez..."):
                embeddings = OllamaEmbeddings(model=EMBED_MODEL)
                progress_bar.progress(20, text="📚 Cargando archivo...")
                
                loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
                documents = loader.load()
                progress_bar.progress(40, text="📚 Dividiendo en fragmentos...")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks = text_splitter.split_documents(documents)
                progress_bar.progress(60, text=f"📚 Generando {len(chunks)} fragmentos...")

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                )
                progress_bar.progress(80, text="📚 Guardando en base de datos...")
                
                vectorstore.persist()
                progress_bar.progress(100, text="✅ Procesamiento completado!")
                time.sleep(0.5)
                progress_bar.empty()
                
                st.success(f"✅ Archivo procesado: {len(chunks)} fragmentos")
                return vectorstore
    except Exception as e:
        st.error(f"❌ Error al crear/cargar vectorstore: {str(e)}")
        return None

# ============================================
# CARGA DE RETRIEVER (RAG) BASE
# ============================================
@st.cache_resource
def cargar_retriever():
    """Devuelve un retriever basado en la base vectorial, o None si falla."""
    vectorstore = crear_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 20})

# ============================================
# FUNCIÓN PARA CONSTRUIR MENSAJES CON HISTORIAL
# ============================================
def construir_mensajes_con_historial(pregunta, docs=None, datos_numericos=None):
    """
    Construye la lista de mensajes para el modelo,
    incluyendo system prompt, historial y contexto (RAG y numérico).
    """
    mensajes = [SystemMessage(content=SYSTEM_PROMPT)]

    # Añadir historial reciente de la conversación
    for msg in st.session_state.messages[-MAX_HISTORY:]:
        if msg["role"] == "user":
            mensajes.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            mensajes.append(AIMessage(content=msg["content"]))

    # Preparar contexto completo
    contexto_completo = []
    
    # Añadir contexto de RAG si hay documentos
    if docs:
        contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        contexto_completo.append(f"CONTEXTO DOCUMENTAL:\n{contexto_rag}")
    
    # Añadir análisis numérico si hay datos
    if datos_numericos and isinstance(datos_numericos, DatosNumericos):
        resumen_numerico = datos_numericos.generar_resumen_estadistico()
        if resumen_numerico:
            contexto_completo.append(f"ANÁLISIS NUMÉRICO:\n{resumen_numerico}")
    
    # Construir mensaje final
    if contexto_completo:
        mensaje_usuario = f"{chr(10).join(contexto_completo)}\n\nPREGUNTA: {pregunta}"
    else:
        mensaje_usuario = pregunta
    
    mensajes.append(HumanMessage(content=mensaje_usuario))

    return mensajes

# ============================================
# FUNCIÓN PARA APLICAR ESTILOS
# ============================================
def set_background(image_path: str):
    """Aplica fondo y tema visual personalizado si existe la imagen."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
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
            .respuesta-aguwey h2 {{
                font-size: 1.5rem;
                border-bottom: 1px solid var(--border);
                padding-bottom: 0.3rem;
            }}
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
            """,
            unsafe_allow_html=True,
        )

# ============================================
# STREAMING DE RESPUESTA
# ============================================
def mostrar_respuesta_streaming(mensajes):
    """Gestiona el streaming de la respuesta del modelo en la interfaz."""
    try:
        response_container = st.empty()
        callback = StreamlitCallbackHandler(response_container)

        llm_stream = ChatOllama(
            model=MODEL_NAME,
            temperature=0.0,
            num_ctx=4096,
            top_p=0.9,
            repeat_penalty=1.3,
            streaming=True,
            callbacks=[callback],
        )

        response = llm_stream.invoke(mensajes)

        # Mostrar respuesta final sin cursor
        response_container.markdown(
            f'<div class="respuesta-aguwey">{response.content}</div>',
            unsafe_allow_html=True,
        )
        return response.content

    except Exception as e:
        st.error(f"❌ Error en streaming: {str(e)}")
        return None

# ============================================
# FUNCIÓN MEJORADA: EXTRAER TEXTO DE CUALQUIER DOCUMENTO
# CON SOPORTE NUMÉRICO Y MÚLTIPLES FORMATOS
# ============================================
def extraer_texto_de_archivo(uploaded_file):
    """
    Versión mejorada que extrae texto y datos numéricos estructurados
    """
    nombre = uploaded_file.name.lower()
    datos_numericos = DatosNumericos()
    datos_numericos.fuente = uploaded_file.name
    datos_numericos.tipo_archivo = nombre.split('.')[-1]
    
    # PDF
    if nombre.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        texto_completo = []
        for page in reader.pages:
            texto = page.extract_text() or ""
            texto_completo.append(texto)
        
        texto_final = "\n".join(texto_completo)
        datos_numericos.extraer_numeros_texto(texto_final)
        return texto_final, datos_numericos

    # Word (.docx)
    if nombre.endswith(".docx"):
        doc = Document(uploaded_file)
        texto_completo = [p.text for p in doc.paragraphs if p.text.strip()]
        texto_final = "\n".join(texto_completo)
        datos_numericos.extraer_numeros_texto(texto_final)
        return texto_final, datos_numericos

    # TXT - con detección de codificación
    if nombre.endswith(".txt"):
        contenido = uploaded_file.read()
        try:
            texto_final = contenido.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            # Detectar codificación automáticamente
            encoding = chardet.detect(contenido)['encoding'] or 'latin-1'
            texto_final = contenido.decode(encoding, errors="ignore")
        
        datos_numericos.extraer_numeros_texto(texto_final)
        return texto_final, datos_numericos

    # Excel - ANÁLISIS NUMÉRICO MEJORADO
    if nombre.endswith((".xlsx", ".xls")):
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            textos_hojas = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                textos_hojas.append(f"\n--- Hoja: {sheet_name} ---")
                textos_hojas.append(df.to_string())
                
                # Guardar DataFrame para análisis numérico
                datos_numericos.agregar_dataframe(sheet_name, df)
            
            texto_final = "\n\n".join(textos_hojas)
            return texto_final, datos_numericos
            
        except Exception as e:
            return f"No se pudo leer el Excel: {str(e)}", None

    # CSV - NUEVO SOPORTE
    if nombre.endswith(".csv"):
        try:
            # Leer CSV con detección automática
            contenido = uploaded_file.read()
            encoding = chardet.detect(contenido)['encoding'] or 'utf-8'
            
            # Intentar diferentes separadores
            contenido_str = contenido.decode(encoding, errors='ignore')
            first_line = contenido_str.split('\n')[0]
            
            separadores = [',', ';', '\t', '|', ':']
            separador_usado = ','
            
            for sep in separadores:
                if sep in first_line:
                    separador_usado = sep
                    break
            
            # Leer CSV
            df = pd.read_csv(io.StringIO(contenido_str), sep=separador_usado)
            
            # Guardar en datos numéricos
            datos_numericos.agregar_dataframe("CSV", df)
            
            # Generar texto
            texto_final = f"--- Archivo CSV: {uploaded_file.name} ---\n"
            texto_final += f"Separador: '{separador_usado}'\n"
            texto_final += f"Filas: {len(df)}, Columnas: {len(df.columns)}\n\n"
            texto_final += df.to_string()
            
            return texto_final, datos_numericos
            
        except Exception as e:
            return f"No se pudo leer el CSV: {str(e)}", None

    # JSON - NUEVO SOPORTE
    if nombre.endswith(".json"):
        try:
            contenido = uploaded_file.read()
            data = json.loads(contenido)
            
            # Intentar convertir a DataFrame si es posible
            try:
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                    datos_numericos.agregar_dataframe("JSON", df)
                    texto_final = df.to_string()
                else:
                    texto_final = json.dumps(data, indent=2, ensure_ascii=False)
                    # Extraer números del JSON
                    numeros = re.findall(r'-?\d+\.?\d*', texto_final)
                    datos_numericos.numeros_extraidos = [float(n) for n in numeros]
            except:
                texto_final = json.dumps(data, indent=2, ensure_ascii=False)
            
            return texto_final, datos_numericos
            
        except Exception as e:
            return f"No se pudo leer el JSON: {str(e)}", None

    # XML - NUEVO SOPORTE (básico)
    if nombre.endswith(".xml"):
        try:
            contenido = uploaded_file.read()
            encoding = chardet.detect(contenido)['encoding'] or 'utf-8'
            texto_final = contenido.decode(encoding, errors='ignore')
            
            # Extraer números del XML
            datos_numericos.extraer_numeros_texto(texto_final)
            return texto_final, datos_numericos
            
        except Exception as e:
            return f"No se pudo leer el XML: {str(e)}", None

    # Imágenes (OCR)
    if nombre.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
        try:
            image = Image.open(uploaded_file)
            texto_final = pytesseract.image_to_string(image, lang="spa+eng")
            datos_numericos.extraer_numeros_texto(texto_final)
            return texto_final if texto_final.strip() else "No se detectó texto en la imagen.", datos_numericos
        except Exception as e:
            return f"No se pudo procesar la imagen: {str(e)}", None

    return "Tipo de archivo no soportado actualmente.", None

# ============================================
# VECTORSTORE TEMPORAL DESDE TEXTO (DOCUMENTO)
# ============================================
def crear_vectorstore_desde_texto(texto):
    try:
        progress_bar = st.progress(0, text="📄 Preparando documento...")
        
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        progress_bar.progress(30, text="📄 Dividiendo documento en fragmentos...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = text_splitter.create_documents([texto])
        progress_bar.progress(60, text=f"📄 Generando {len(docs)} fragmentos...")
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        progress_bar.progress(100, text="✅ Documento indexado correctamente!")
        time.sleep(0.5)
        progress_bar.empty()
        
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error al crear vectorstore temporal: {str(e)}")
        return None

# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def main():
    st.set_page_config(
        page_title="AguweyBot PRO",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    set_background("fondo.png")

    # Inicializar session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "first_interaction" not in st.session_state:
        st.session_state.first_interaction = True
    if "doc_vectorstore" not in st.session_state:
        st.session_state.doc_vectorstore = None
    if "doc_nombre" not in st.session_state:
        st.session_state.doc_nombre = None
    if "usar_doc_en_analisis" not in st.session_state:
        st.session_state.usar_doc_en_analisis = False
    if "datos_numericos" not in st.session_state:  # NUEVO
        st.session_state.datos_numericos = None

    # Sidebar
    with st.sidebar:
        st.markdown("# ⚡AGUWEYBOT")
        st.markdown("### *Asistente Inteligente*")

        if os.path.exists("logo.png"):
            st.image("logo.png", width=220)
        else:
            st.info("📌 Coloca 'logo.png' para personalizar")

        st.markdown("---")

        # Información del conocimiento base
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
                st.session_state.doc_vectorstore = None
                st.session_state.doc_nombre = None
                st.session_state.usar_doc_en_analisis = False
                st.session_state.datos_numericos = None
                st.rerun()

        st.markdown("---")

        # Análisis de documentos (MEJORADO)
        st.markdown("### 📎 Análisis de Documentos")
        st.markdown("**Formatos soportados:**")
        st.markdown("📊 Excel, CSV, JSON, XML")
        st.markdown("📄 PDF, Word, TXT")
        st.markdown("🖼️ Imágenes (OCR)")
        
        uploaded_file = st.file_uploader(
            "Sube un archivo",
            type=["pdf", "docx", "txt", "xlsx", "xls", "csv", "json", "xml", 
                  "png", "jpg", "jpeg", "bmp", "tiff"],
        )

        if uploaded_file is not None:
            st.success(f"📄 Archivo cargado: **{uploaded_file.name}**")

            usar_doc_en_analisis = st.checkbox(
                "Usar este documento como contexto principal (RAG)",
                value=True,
                key="usar_doc_en_analisis",
            )

            if st.button("🔍 Procesar documento"):
                progress_bar = st.progress(0, text="📄 Iniciando extracción de texto...")
                
                texto_doc, datos_numericos = extraer_texto_de_archivo(uploaded_file)
                progress_bar.progress(50, text="📄 Texto extraído, indexando...")
                
                if texto_doc and len(texto_doc.strip()) > 0:
                    vectorstore_doc = crear_vectorstore_desde_texto(texto_doc)
                    if vectorstore_doc:
                        st.session_state.doc_vectorstore = vectorstore_doc
                        st.session_state.doc_nombre = uploaded_file.name
                        st.session_state.datos_numericos = datos_numericos
                        progress_bar.progress(100, text="✅ Documento listo para consultas!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        
                        # Mostrar resumen numérico si existe
                        if datos_numericos and (datos_numericos.dataframes or datos_numericos.numeros_extraidos):
                            st.success("📊 **Datos numéricos detectados!**")
                            with st.expander("Ver resumen estadístico"):
                                st.markdown(datos_numericos.generar_resumen_estadistico())
                else:
                    progress_bar.empty()
                    st.warning("⚠️ No se obtuvo texto útil del archivo.")
        else:
            st.session_state.usar_doc_en_analisis = False

        st.markdown("---")
        st.markdown("### 🎯 Capacidades")
        st.markdown(
            """
- 📚 RAG con conocimiento.txt
- 📎 RAG con documentos cargados
- 🔬 Análisis Técnico
- ✍️ Escritura Creativa
- 📊 Análisis de Datos Numéricos
- 💬 Memoria de Conversación
- ⚡ Streaming Avanzado
            """
        )

        st.markdown("---")
        st.markdown("### 📊 Estado del Sistema")
        st.success(f"✅ Modelo: {MODEL_NAME}")
        st.info(f"📁 Memoria: {len(st.session_state.messages)} mensajes")
        st.success("⚡ Streaming: Activado")
        if st.session_state.datos_numericos:
            if st.session_state.datos_numericos.dataframes:
                st.info(f"📊 DataFrames: {len(st.session_state.datos_numericos.dataframes)}")
            if st.session_state.datos_numericos.numeros_extraidos:
                st.info(f"🔢 Valores numéricos: {len(st.session_state.datos_numericos.numeros_extraidos)}")
        st.markdown("---")
        st.caption("CC-NC-SA: 2026 AguweyBot PRO")

    # Contenido principal
    st.markdown(
        '<h1 style="color: #00ffff; text-shadow: 0 0 20px rgba(0,255,255,0.5);">⚡ AguweyBot PRO</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Sistema cognitivo con memoria de conversación y recuperación semántica")

    # Cargar retriever base (conocimiento.txt)
    with st.spinner("🚀 Inicializando sistemas cognitivos..."):
        retriever_base = cargar_retriever()
        if retriever_base is None and os.path.exists(KNOWLEDGE_FILE):
            st.warning(
                "⚠️ No se pudo inicializar la base vectorial. Se responderá solo con el modelo base (sin RAG)."
            )

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(
                    f'<div class="respuesta-aguwey">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(message["content"])

    # Entrada del usuario
    prompt = st.chat_input("Escribe tu consulta...")
    if prompt:
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)

        # Guardar en historial
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Decidir qué retriever usar
        retriever_actual = None
        docs = []

        if st.session_state.usar_doc_en_analisis and st.session_state.doc_vectorstore:
            # Se usará el documento como contexto principal
            retriever_actual = st.session_state.doc_vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
        elif retriever_base:
            # Uso del conocimiento base general
            retriever_actual = retriever_base

        # Buscar documentos con RAG (si disponible)
        with st.spinner("🧠 Pensando..."):
            if retriever_actual:
                docs = retriever_actual.invoke(prompt)
            else:
                docs = []

        # Construir mensajes para el modelo (ahora incluye datos numéricos)
        mensajes = construir_mensajes_con_historial(
            prompt, 
            docs if docs else None,
            st.session_state.datos_numericos if st.session_state.usar_doc_en_analisis else None
        )

        # Generar respuesta con streaming
        with st.chat_message("assistant"):
            respuesta = mostrar_respuesta_streaming(mensajes)

        # Guardar respuesta en historial
        if respuesta:
            st.session_state.messages.append(
                {"role": "assistant", "content": respuesta}
            )

        # Mostrar fuentes consultadas (si hubo RAG)
        if docs:
            with st.expander("📚 Fuentes consultadas"):
                if st.session_state.doc_nombre:
                    st.markdown(f"**Documento:** {st.session_state.doc_nombre}")
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Fragmento {i}:**")
                    st.caption(doc.page_content[:400] + "...")

    # Mensaje de bienvenida inicial
    elif st.session_state.first_interaction:
        st.info(
            "👋 **¡Bienvenido!** "
            "Puedes hacerme preguntas técnicas o literarias. "
            "También puedes subir documentos (PDF, Word, Excel, CSV, JSON, XML, imágenes) para analizarlos. "
            "**¡NUEVO!** Ahora puedo analizar datos numéricos y generar estadísticas automáticamente. "
            "Recuerdo toda la conversación, así que podemos profundizar en temas complejos."
        )
        st.session_state.first_interaction = False

    # Footer
    st.markdown(
        """
        <div class="fixed-footer">
        <strong>⚡ Licencia CC-NC-SA</strong> • Prof. Raymond Rosa Ávila • AguweyBot PRO 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================
# MODULOS AVANZADOS DE ANALISIS DE DOCUMENTOS
# ============================================

def analizar_documento_completo(vectorstore, pregunta):

    try:
        docs = vectorstore.similarity_search(pregunta, k=25)

        texto = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Analiza el documento completo.

PREGUNTA:
{pregunta}

DOCUMENTO:
{texto}
"""

        return prompt

    except Exception as e:
        return f"Error en análisis completo: {e}"


def resumir_documento(vectorstore):

    try:

        docs = vectorstore.similarity_search("", k=30)

        texto = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Resume el documento:

1 Tema principal
2 Ideas clave
3 Resultados importantes
4 Conclusiones

DOCUMENTO:
{texto}
"""

        return prompt

    except Exception as e:
        return f"Error al resumir: {e}"


def extraer_conocimiento(vectorstore):

    try:

        docs = vectorstore.similarity_search("", k=30)

        texto = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Extrae del documento:

- conceptos clave
- definiciones
- ecuaciones
- descubrimientos
- aplicaciones

DOCUMENTO:
{texto}
"""

        return prompt

    except Exception as e:
        return f"Error extrayendo conocimiento: {e}"
# ============================================
# HIERARCHICAL RAG PARA DOCUMENTOS MUY LARGOS
# ============================================

def generar_resumenes_por_fragmento(vectorstore):

    """
    Crea mini-resúmenes de cada fragmento del documento.
    Ideal para libros o papers largos.
    """

    try:

        docs = vectorstore.similarity_search("", k=50)

        resumenes = []

        for i, doc in enumerate(docs):

            fragmento = doc.page_content

            prompt = f"""
Resume el siguiente fragmento del documento en 3-5 líneas.

FRAGMENTO:
{fragmento}
"""

            resumenes.append(prompt)

        return resumenes

    except Exception as e:

        return f"Error generando resúmenes: {e}"


def generar_meta_resumen(vectorstore):

    """
    Genera un resumen global del documento completo
    usando la técnica hierarchical RAG.
    """

    try:

        docs = vectorstore.similarity_search("", k=40)

        texto = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
A partir del siguiente contenido genera un resumen profundo del documento.

Incluye:

- tema central
- estructura del documento
- argumentos principales
- conclusiones

DOCUMENTO:
{texto}
"""

        return prompt

    except Exception as e:

        return f"Error generando meta resumen: {e}"


def responder_con_contexto_amplio(vectorstore, pregunta):

    """
    Usa muchos fragmentos para responder preguntas
    complejas sobre documentos largos.
    """

    try:

        docs = vectorstore.similarity_search(pregunta, k=30)

        contexto = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Usa el siguiente contexto para responder la pregunta.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}
"""

        return prompt

    except Exception as e:

        return f"Error analizando documento largo: {e}"
if __name__ == "__main__":
    main()