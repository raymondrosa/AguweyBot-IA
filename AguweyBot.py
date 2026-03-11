# ============================================
# AGUWEYBOT PRO - CON MEMORIA, RAG Y ANÁLISIS DE DOCUMENTOS
# VERSIÓN MEJORADA CON ANÁLISIS NUMÉRICO Y DE CÓDIGO
# VERSIÓN OPTIMIZADA CON FUNCIONALIDAD DE AUDIO (ACCESIBILIDAD)
# VERSIÓN CORREGIDA - BOTONES DE AUDIO PERSISTENTES
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
# NUEVOS IMPORTS PARA FUNCIONALIDAD DE AUDIO
# ============================================
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("⚠️ Instala gtts para funcionalidad de texto a voz: pip install gtts")

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    st.warning("⚠️ Instala SpeechRecognition para entrada por voz: pip install SpeechRecognition")

# ============================================
# CONFIGURACIÓN (SIN CAMBIOS)
# ============================================
MODEL_NAME = "phi3:mini"
EMBED_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "vector_db"
KNOWLEDGE_FILE = "conocimiento.txt"
MAX_HISTORY = 10

# ============================================
# SYSTEM PROMPT (MEJORADO CON REGLAS DE PRECISIÓN)
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
- **NO te limites a listar números, busca relaciones y significado**
- **Proporciona insights útiles basados en los datos**

TAMBIÉN ERES EXPERTO EN ANÁLISIS DE CÓDIGO:
- Identifica el lenguaje de programación
- Explica la estructura y lógica del código
- Señala posibles errores o mejoras
- Sugiere optimizaciones
- Describe qué hace cada función o clase
- Ayuda a depurar y entender código complejo

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

MODO ANÁLISIS DE CÓDIGO:
- Analiza la estructura del código
- Explica la lógica y funcionamiento
- Identifica posibles bugs o mejoras
- Sugiere optimizaciones
- Documenta funciones y clases

REGLAS GENERALES:
- Detecta automáticamente si la consulta es técnica, creativa, de análisis de datos o de código.
- Si contiene tablas o números → activa modo análisis de datos.
- Si contiene código fuente → activa modo análisis de código.
- Nunca rechaces ayudar en escritura creativa.
- No menciones limitaciones innecesarias.
- Mantén coherencia y calidad en todos los modos.
- Si falta información, pide aclaración de forma profesional.

REGLAS DE PRECISIÓN Y HONESTIDAD:
- **REGLAS DE ORO:**
    1.  **NO INVENTES INFORMACIÓN.** Si no sabes la respuesta o si la información requerida no está presente en el contexto proporcionado (RAG, datos numéricos, código) o en tu conocimiento base fundamental, indícalo claramente.
    2.  Es preferible decir "No encuentro información sobre eso en los documentos proporcionados" o "Eso está fuera de mi conocimiento actual", a ofrecer una respuesta especulativa o incorrecta.
- Cuando utilices información proveniente del contexto RAG o de los datos numéricos, prefija la oración con frases como "Según el documento...", "Los datos indican...", "El análisis del código muestra...". Esto te ayudará a mantenerte anclado a las fuentes.
- Si la pregunta es ambigua, pide aclaración en lugar de asumir la intención del usuario.
- Antes de responder, verifica mentalmente que tu respuesta esté directamente respaldada por la información disponible.

DIRECTRICES DE ESTILO:
- Utiliza emojis estratégicamente para mejorar la comunicación visual
- En modo técnico: usa emojis para secciones (🔬, 📊, ⚙️, 📐)
- En modo creativo: usa emojis expresivos (✍️, 📖, 🎭, ✨)
- En modo análisis de datos: usa (📈, 📉, 📊, 🔢, 📋)
- En modo análisis de código: usa (💻, 🔧, 🐛, ⚡, 📝)
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
💻 Análisis de Código:
🔧 Optimización sugerida:
🐛 Posible error:
⚡ Mejora de rendimiento:

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
        self.codigo_detectado = False
        self.fragmentos_codigo = []
    
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
    
    def detectar_codigo(self, texto):
        """Detecta si el texto contiene código fuente"""
        patrones_codigo = [
            r'def\s+\w+\s*\([^)]*\)\s*:',
            r'class\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'if\s+__name__\s*==\s*["\']__main__["\']',
            r'function\s+\w+\s*\([^)]*\)\s*{',
            r'public\s+class\s+\w+',
            r'#include\s+<[^>]+>',
            r'package\s+\w+',
            r'namespace\s+\w+',
            r'<\?php',
            r'console\.log\(',
            r'System\.out\.println',
            r'printf\('
        ]
        
        for patron in patrones_codigo:
            if re.search(patron, texto, re.MULTILINE):
                self.codigo_detectado = True
                # Extraer fragmentos de código (líneas con indentación)
                lineas = texto.split('\n')
                fragmento_actual = []
                en_codigo = False
                
                for linea in lineas:
                    if linea.startswith(('    ', '\t')) or any(p in linea for p in ['def ', 'class ', 'if ', 'for ', 'while ']):
                        en_codigo = True
                        fragmento_actual.append(linea)
                    elif en_codigo and linea.strip() == '':
                        fragmento_actual.append(linea)
                    elif en_codigo and not linea.startswith(('    ', '\t')):
                        if fragmento_actual:
                            self.fragmentos_codigo.append('\n'.join(fragmento_actual))
                            fragmento_actual = []
                        en_codigo = False
                
                if fragmento_actual:
                    self.fragmentos_codigo.append('\n'.join(fragmento_actual))
                
                return True
        return False
    
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
        
        if self.codigo_detectado:
            resumen.append("\n💻 **CÓDIGO FUENTE DETECTADO**")
            resumen.append(f"Fragmentos de código: {len(self.fragmentos_codigo)}")
        
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
# FUNCIÓN PARA DETECTAR TIPO DE CONSULTA
# ============================================
def detectar_tipo_consulta(pregunta, datos_numericos=None):
    """Detecta si la consulta es sobre datos numéricos o código"""
    
    # Palabras clave para análisis numérico
    keywords_numericos = ['promedio', 'media', 'mediana', 'suma', 'total', 
                          'estadística', 'tendencia', 'gráfico', 'máximo', 
                          'mínimo', 'correlación', 'porcentaje', 'distribución',
                          'estadistico', 'variacion', 'desviacion', 'varianza',
                          'cuartil', 'percentil', 'histograma', 'boxplot']
    
    # Palabras clave para código
    keywords_codigo = ['código', 'función', 'clase', 'método', 'programa',
                       'algoritmo', 'debug', 'error', 'lógica', 'script',
                       'python', 'java', 'javascript', 'html', 'css',
                       'compilar', 'ejecutar', 'variable', 'bucle', 'condicion',
                       'import', 'def', 'class', 'function', 'console',
                       'print', 'return', 'if', 'else', 'for', 'while']
    
    pregunta_lower = pregunta.lower()
    
    # Verificar si hay datos numéricos disponibles
    tiene_datos = (datos_numericos and 
                  (datos_numericos.dataframes or datos_numericos.numeros_extraidos))
    
    # Verificar si hay código disponible
    tiene_codigo = (datos_numericos and datos_numericos.codigo_detectado)
    
    # Detectar tipo
    if tiene_datos and any(keyword in pregunta_lower for keyword in keywords_numericos):
        return "numerico"
    elif (tiene_codigo or any(keyword in pregunta_lower for keyword in keywords_codigo)):
        return "codigo"
    else:
        return "general"

# ============================================
# FUNCIONES DE ACCESIBILIDAD POR AUDIO (VERSIÓN MEJORADA)
# ============================================

def texto_a_voz(texto, idioma="es"):
    """Convierte texto a audio y devuelve HTML con reproductor"""
    if not TTS_AVAILABLE:
        return "<p style='color: #f85149;'>⚠️ gTTS no instalado. Ejecuta: pip install gtts</p>", None
    
    try:
        tts = gTTS(text=texto, lang=idioma, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        
        audio_html = f"""
        <audio id="audio-player-{int(time.time())}" controls style="width: 100%; margin: 10px 0; border-radius: 30px;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Tu navegador no soporta el elemento de audio.
        </audio>
        """
        return audio_html, audio_base64
    except Exception as e:
        return f"<p style='color: #f85149;'>❌ Error generando audio: {e}</p>", None

# ============================================
# FUNCIÓN PARA CREAR CONTROLES DE AUDIO PERSISTENTES
# ============================================
def crear_controles_audio(respuesta, msg_id):
    """
    Crea controles de audio que no desaparecen al hacer clic
    """
    # Inicializar el estado de audio si no existe
    if "audio_states" not in st.session_state:
        st.session_state.audio_states = {}
    
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    
    # Estado para este mensaje específico
    if msg_id not in st.session_state.audio_states:
        st.session_state.audio_states[msg_id] = {
            "mostrar_reproductor": False,
            "audio_generado": False
        }
    
    msg_state = st.session_state.audio_states[msg_id]
    
    # Contenedor para los botones (siempre visibles)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
    
    with col1:
        # Botón para escuchar/reproducir
        if st.button("🔊 Escuchar", key=f"btn_audio_{msg_id}"):
            # Cambiar el estado para mostrar el reproductor
            msg_state["mostrar_reproductor"] = True
            msg_state["audio_generado"] = False
            st.session_state.audio_states[msg_id] = msg_state
            st.rerun()
    
    with col2:
        if st.button("📋 Copiar", key=f"btn_copy_{msg_id}"):
            # Usar JavaScript para copiar al portapapeles
            js_code = f"""
            <script>
                navigator.clipboard.writeText({repr(respuesta)});
                document.getElementById('copy-status-{msg_id}').style.display = 'block';
                setTimeout(function() {{
                    document.getElementById('copy-status-{msg_id}').style.display = 'none';
                }}, 2000);
            </script>
            <div id="copy-status-{msg_id}" style="color: #00ff00; margin: 5px 0; display: none;">✅ ¡Copiado!</div>
            """
            st.markdown(js_code, unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Regenerar", key=f"btn_regenerate_{msg_id}"):
            # Forzar regeneración del audio
            if msg_id in st.session_state.audio_cache:
                del st.session_state.audio_cache[msg_id]
            msg_state["mostrar_reproductor"] = True
            msg_state["audio_generado"] = False
            st.session_state.audio_states[msg_id] = msg_state
            st.rerun()
    
    # Contenedor para el reproductor de audio (visible solo cuando se solicita)
    if msg_state["mostrar_reproductor"]:
        audio_placeholder = st.empty()
        with audio_placeholder:
            # Generar o recuperar audio de caché
            if msg_id not in st.session_state.audio_cache:
                audio_html, audio_base64 = texto_a_voz(respuesta)
                if audio_base64:
                    st.session_state.audio_cache[msg_id] = audio_base64
                    msg_state["audio_generado"] = True
                    st.session_state.audio_states[msg_id] = msg_state
            
            if msg_id in st.session_state.audio_cache:
                audio_base64 = st.session_state.audio_cache[msg_id]
                # Botón para cerrar el reproductor
                close_button = f"""
                <div style="display: flex; justify-content: flex-end; margin-top: 5px;">
                    <button onclick="(function(){{ 
                        var audioDiv = document.getElementById('audio-container-{msg_id}');
                        if(audioDiv) audioDiv.style.display = 'none';
                    }})();" 
                            style="background: transparent; border: 1px solid #00ffff; color: #00ffff; border-radius: 15px; padding: 2px 10px; cursor: pointer; margin-bottom: 5px;">
                        ✖ Cerrar reproductor
                    </button>
                </div>
                """
                
                audio_html = f"""
                <div id="audio-container-{msg_id}" style="margin-top: 10px; padding: 10px; background: rgba(0,255,255,0.1); border-radius: 10px;">
                    {close_button}
                    <audio controls autoplay style="width: 100%; border-radius: 30px;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                </div>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

# ============================================
# CALLBACK PARA STREAMING (OPTIMIZADO)
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
        # Delay reducido para más velocidad
        time.sleep(0.002)

# ============================================
# STREAMING DE RESPUESTA (PARÁMETROS OPTIMIZADOS)
# ============================================
def mostrar_respuesta_streaming(mensajes):
    """Gestiona el streaming de la respuesta del modelo en la interfaz."""
    try:
        response_container = st.empty()
        callback = StreamlitCallbackHandler(response_container)

        llm_stream = ChatOllama(
            model=MODEL_NAME,
            temperature=0.1,  # Optimizado para precisión
            num_ctx=4096,
            top_p=0.85,       # Optimizado para precisión
            repeat_penalty=1.2, # Optimizado para evitar repeticiones
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
# FUNCIÓN PARA CONSTRUIR MENSAJES CON HISTORIAL (VERSIÓN SIMPLIFICADA)
# ============================================
def construir_mensajes_con_historial(pregunta, docs=None, datos_numericos=None):
    """
    Versión simplificada que asegura que el contexto se incluya
    """
    mensajes = [SystemMessage(content=SYSTEM_PROMPT)]

    # Añadir historial reciente
    for msg in st.session_state.messages[-MAX_HISTORY:]:
        if msg["role"] == "user":
            mensajes.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            mensajes.append(AIMessage(content=msg["content"]))

    # Preparar contexto del documento
    contexto = ""
    
    if docs and len(docs) > 0:
        st.success(f"📚 Usando {len(docs)} fragmentos del documento")
        
        # Construir contexto con todos los fragmentos
        fragmentos = []
        for i, doc in enumerate(docs):
            contenido = doc.page_content.strip()
            if contenido:
                fragmentos.append(f"--- FRAGMENTO {i+1} ---\n{contenido}")
        
        if fragmentos:
            contexto = "CONTEXTO DEL DOCUMENTO:\n\n" + "\n\n".join(fragmentos)
            st.success("✅ Contexto preparado correctamente")
    else:
        st.warning("⚠️ No hay fragmentos para usar como contexto")
        contexto = "No se encontró información relevante en el documento."

    # Instrucción clara
    instruccion = """
    IMPORTANTE: Responde la pregunta del usuario usando ÚNICAMENTE la información del CONTEXTO DEL DOCUMENTO proporcionado arriba.
    Si la respuesta no está en el contexto, di honestamente que no encuentras esa información en el documento.
    No inventes información ni uses conocimiento externo.
    """
    
    # Construir mensaje completo
    mensaje_usuario = f"{contexto}\n\n{instruccion}\n\nPREGUNTA: {pregunta}\n\nRESPUESTA (basada en el contexto):"
    
    mensajes.append(HumanMessage(content=mensaje_usuario))
    return mensajes
# ============================================
# FUNCIÓN PARA APLICAR ESTILOS (COMPLETA)
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
# FUNCIÓN CORREGIDA: EXTRAER TEXTO DE CUALQUIER DOCUMENTO
# ============================================
def extraer_texto_de_archivo(uploaded_file):
    """
    Versión mejorada que extrae texto, datos numéricos estructurados y código
    VERSIÓN OPTIMIZADA Y CORREGIDA: Maneja correctamente múltiples lecturas del archivo
    """
    nombre = uploaded_file.name.lower()
    datos_numericos = DatosNumericos()
    datos_numericos.fuente = uploaded_file.name
    datos_numericos.tipo_archivo = nombre.split('.')[-1]
    
    # Guardar el contenido original para archivos que necesitan múltiples lecturas
    file_content = uploaded_file.getvalue()
    
    # Mostrar información de depuración
    st.info(f"📄 Procesando archivo: {uploaded_file.name} ({len(file_content)} bytes)")
    
    # PDF - optimizado
    if nombre.endswith(".pdf"):
        try:
            # Crear un nuevo objeto BytesIO con el contenido guardado
            pdf_file = io.BytesIO(file_content)
            reader = PdfReader(pdf_file)
            texto_completo = []
            # Limitar a primeras 20 páginas para mejor cobertura
            paginas = min(20, len(reader.pages))
            for i in range(paginas):
                texto = reader.pages[i].extract_text() or ""
                if texto.strip():
                    texto_completo.append(f"[Página {i+1}]:\n{texto}")
            
            if texto_completo:
                texto_final = "\n\n".join(texto_completo)
                st.success(f"✅ PDF procesado: {len(texto_completo)} páginas")
            else:
                texto_final = "No se pudo extraer texto del PDF"
                st.warning("⚠️ No se pudo extraer texto del PDF")
            
            datos_numericos.extraer_numeros_texto(texto_final)
            datos_numericos.detectar_codigo(texto_final)
            return texto_final, datos_numericos
        except Exception as e:
            st.error(f"Error procesando PDF: {str(e)}")
            return f"Error procesando PDF: {str(e)}", datos_numericos

    # Word (.docx) - optimizado
    if nombre.endswith(".docx"):
        try:
            # Crear un nuevo objeto BytesIO con el contenido guardado
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            # Limitar a primeros 100 párrafos
            textos = []
            for i, p in enumerate(doc.paragraphs[:100]):
                if p.text.strip():
                    textos.append(f"[Párrafo {i+1}]:\n{p.text}")
            
            if textos:
                texto_final = "\n\n".join(textos)
                st.success(f"✅ Word procesado: {len(textos)} párrafos")
            else:
                texto_final = "No se pudo extraer texto del documento Word"
                st.warning("⚠️ No se pudo extraer texto del documento Word")
            
            datos_numericos.extraer_numeros_texto(texto_final)
            datos_numericos.detectar_codigo(texto_final)
            return texto_final, datos_numericos
        except Exception as e:
            st.error(f"Error procesando Word: {str(e)}")
            return f"Error procesando Word: {str(e)}", datos_numericos

    # TXT - con detección de codificación
    if nombre.endswith(".txt"):
        try:
            try:
                texto_final = file_content.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                # Detectar codificación automáticamente
                encoding = chardet.detect(file_content)['encoding'] or 'latin-1'
                texto_final = file_content.decode(encoding, errors="ignore")
            
            if texto_final.strip():
                st.success(f"✅ TXT procesado: {len(texto_final)} caracteres")
            else:
                texto_final = "El archivo TXT está vacío"
                st.warning("⚠️ El archivo TXT está vacío")
            
            datos_numericos.extraer_numeros_texto(texto_final)
            datos_numericos.detectar_codigo(texto_final)
            return texto_final, datos_numericos
        except Exception as e:
            st.error(f"Error procesando TXT: {str(e)}")
            return f"Error procesando TXT: {str(e)}", datos_numericos

    # Excel - ANÁLISIS NUMÉRICO (CORREGIDO)
    if nombre.endswith((".xlsx", ".xls")):
        try:
            # Crear un nuevo objeto BytesIO para el archivo Excel
            excel_file = pd.ExcelFile(io.BytesIO(file_content))
            textos_hojas = []
            
            # Limitar a primeras 5 hojas
            for sheet_name in excel_file.sheet_names[:5]:
                # Crear un nuevo BytesIO para cada hoja
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
                
                # Guardar DataFrame completo para análisis numérico
                datos_numericos.agregar_dataframe(sheet_name, df)
                
                # Para el texto, incluir información detallada
                textos_hojas.append(f"\n=== HOJA: {sheet_name} ===")
                textos_hojas.append(f"Filas: {len(df)}, Columnas: {len(df.columns)}")
                textos_hojas.append(f"Columnas: {', '.join(df.columns.tolist())}")
                
                # Incluir todas las filas si son pocas, o un resumen si son muchas
                if len(df) <= 50:
                    textos_hojas.append("\nDATOS COMPLETOS:")
                    textos_hojas.append(df.to_string())
                else:
                    # Incluir estadísticas descriptivas
                    textos_hojas.append("\nESTADÍSTICAS DESCRIPTIVAS:")
                    textos_hojas.append(df.describe().to_string())
                    
                    # Incluir primeras 20 filas
                    textos_hojas.append("\nPRIMERAS 20 FILAS:")
                    textos_hojas.append(df.head(20).to_string())
                    
                    # Incluir últimas 10 filas
                    textos_hojas.append("\nÚLTIMAS 10 FILAS:")
                    textos_hojas.append(df.tail(10).to_string())
            
            texto_final = "\n\n".join(textos_hojas)
            st.success(f"✅ Excel procesado: {len(excel_file.sheet_names)} hojas")
            return texto_final, datos_numericos
            
        except Exception as e:
            st.error(f"Error procesando Excel: {str(e)}")
            return f"Error procesando Excel: {str(e)}", datos_numericos

    # CSV - VERSIÓN OPTIMIZADA
    if nombre.endswith(".csv"):
        try:
            encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
            
            contenido_str = file_content.decode(encoding, errors='ignore')
            first_line = contenido_str.split('\n')[0]
            
            separadores = [',', ';', '\t', '|', ':']
            separador_usado = ','
            
            for sep in separadores:
                if sep in first_line:
                    separador_usado = sep
                    break
            
            # Leer todo el CSV para mejor análisis
            df = pd.read_csv(io.StringIO(contenido_str), sep=separador_usado)
            
            # Guardar en datos numéricos
            datos_numericos.agregar_dataframe("CSV", df)
            
            # Generar texto enriquecido
            textos = []
            textos.append(f"=== ARCHIVO CSV: {uploaded_file.name} ===")
            textos.append(f"Separador: '{separador_usado}'")
            textos.append(f"Filas totales: {len(df)}")
            textos.append(f"Columnas: {len(df.columns)}")
            textos.append(f"Columnas: {', '.join(df.columns.tolist())}")
            
            textos.append("\nESTADÍSTICAS DESCRIPTIVAS:")
            textos.append(df.describe().to_string())
            
            textos.append("\nPRIMERAS 20 FILAS:")
            textos.append(df.head(20).to_string())
            
            if len(df) > 20:
                textos.append("\nÚLTIMAS 10 FILAS:")
                textos.append(df.tail(10).to_string())
            
            texto_final = "\n\n".join(textos)
            st.success(f"✅ CSV procesado: {len(df)} filas")
            return texto_final, datos_numericos
            
        except Exception as e:
            st.error(f"Error procesando CSV: {str(e)}")
            return f"Error procesando CSV: {str(e)}", datos_numericos

    # JSON - VERSIÓN OPTIMIZADA
    if nombre.endswith(".json"):
        try:
            data = json.loads(file_content)
            
            # Intentar convertir a DataFrame si es posible
            try:
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    datos_numericos.agregar_dataframe("JSON", df)
                    textos = []
                    textos.append("=== DATOS ESTRUCTURADOS JSON ===")
                    textos.append(f"Registros totales: {len(data)}")
                    textos.append(f"Campos: {len(df.columns)}")
                    textos.append(f"Campos: {', '.join(df.columns.tolist())}")
                    
                    textos.append("\nESTADÍSTICAS:")
                    textos.append(df.describe().to_string())
                    
                    textos.append("\nPRIMERAS 20 FILAS:")
                    textos.append(df.head(20).to_string())
                    
                    texto_final = "\n\n".join(textos)
                else:
                    # Si no es tabular, mostrar estructura completa
                    texto_final = json.dumps(data, indent=2, ensure_ascii=False)
                
                st.success(f"✅ JSON procesado")
                return texto_final, datos_numericos
            except:
                texto_final = json.dumps(data, indent=2, ensure_ascii=False)
                return texto_final, datos_numericos
            
        except Exception as e:
            st.error(f"Error procesando JSON: {str(e)}")
            return f"Error procesando JSON: {str(e)}", datos_numericos

    # XML - optimizado
    if nombre.endswith(".xml"):
        try:
            encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
            texto_final = file_content.decode(encoding, errors='ignore')
            st.success(f"✅ XML procesado: {len(texto_final)} caracteres")
            
            datos_numericos.extraer_numeros_texto(texto_final)
            datos_numericos.detectar_codigo(texto_final)
            return texto_final, datos_numericos
            
        except Exception as e:
            st.error(f"Error procesando XML: {str(e)}")
            return f"Error procesando XML: {str(e)}", datos_numericos

    # Imágenes (OCR)
    if nombre.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
        try:
            # Crear un nuevo objeto BytesIO con el contenido guardado
            image_file = io.BytesIO(file_content)
            image = Image.open(image_file)
            texto_final = pytesseract.image_to_string(image, lang="spa+eng")
            
            if texto_final.strip():
                st.success(f"✅ Imagen procesada con OCR: {len(texto_final)} caracteres")
            else:
                texto_final = "No se detectó texto en la imagen."
                st.warning("⚠️ No se detectó texto en la imagen")
            
            datos_numericos.extraer_numeros_texto(texto_final)
            datos_numericos.detectar_codigo(texto_final)
            return texto_final, datos_numericos
        except Exception as e:
            st.error(f"Error procesando imagen: {str(e)}")
            return f"Error procesando imagen: {str(e)}", datos_numericos

    return "Tipo de archivo no soportado actualmente.", datos_numericos

# ============================================
# VECTORSTORE TEMPORAL DESDE TEXTO (VERSIÓN ULTRA SIMPLE)
# ============================================
def crear_vectorstore_desde_texto(texto):
    """Versión simplificada que crea un vectorstore con todos los fragmentos"""
    try:
        # Verificar que hay texto
        if not texto or len(texto.strip()) == 0:
            st.error("❌ No hay texto para procesar")
            return None
            
        st.info(f"📊 Texto a procesar: {len(texto)} caracteres")
        
        # Crear fragmentos más pequeños y numerosos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Fragmentos más pequeños
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = text_splitter.create_documents([texto])
        
        st.info(f"📑 Documento dividido en {len(docs)} fragmentos")
        
        # Mostrar muestra de los primeros fragmentos
        with st.expander("Ver muestra de fragmentos"):
            for i, doc in enumerate(docs[:3]):
                st.text(f"Fragmento {i+1}: {doc.page_content[:200]}...")
        
        # Crear embeddings
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        
        # Crear vectorstore con TODOS los fragmentos
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        
        st.success(f"✅ Vectorstore creado con {len(docs)} fragmentos")
        
        # Verificar que funciona haciendo una búsqueda de prueba
        test_docs = vectorstore.similarity_search("información", k=3)
        if test_docs:
            st.success(f"✅ Prueba de búsqueda exitosa: {len(test_docs)} resultados")
        else:
            st.warning("⚠️ La búsqueda de prueba no encontró resultados")
        
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error al crear vectorstore: {str(e)}")
        return None

# ============================================
# PROCESAR ENTRADA DE VOZ
# ============================================
def procesar_entrada_voz(audio_input):
    """Procesa entrada de audio y devuelve texto"""
    if not STT_AVAILABLE:
        return None, "SpeechRecognition no instalado"
    
    try:
        recognizer = sr.Recognizer()
        
        # Guardar audio temporalmente
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_input.getvalue())
        
        # Cargar y procesar
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
            
        # Intentar reconocer en español
        try:
            texto = recognizer.recognize_google(audio, language="es-ES")
            return texto, None
        except sr.UnknownValueError:
            return None, "No se pudo entender el audio"
        except sr.RequestError:
            return None, "Error en el servicio de reconocimiento"
            
    except Exception as e:
        return None, f"Error procesando audio: {str(e)}"
    finally:
        # Limpiar archivo temporal
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

# ============================================
# MÓDULOS AVANZADOS DE ANÁLISIS DE DOCUMENTOS
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

# ============================================
# INTERFAZ PRINCIPAL (CON FUNCIONALIDAD DE AUDIO MEJORADA)
# ============================================
def main():
    st.set_page_config(
        page_title="AguweyBot PRO - Accesible",
        page_icon="🎧",
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
    if "datos_numericos" not in st.session_state:
        st.session_state.datos_numericos = None
    if "modo_rapido" not in st.session_state:
        st.session_state.modo_rapido = False
    if "auto_leer" not in st.session_state:
        st.session_state.auto_leer = False
    if "procesando_voz" not in st.session_state:
        st.session_state.procesando_voz = False
    if "audio_states" not in st.session_state:
        st.session_state.audio_states = {}
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}

    # Sidebar
    with st.sidebar:
        st.markdown("# 🎧 AGUWEYBOT")
        st.markdown("### *Asistente Inteligente Accesible*")

        if os.path.exists("logo.png"):
            st.image("logo.png", width=220)
        else:
            st.info("📌 Coloca 'logo.png' para personalizar")

        st.markdown("---")

        # ===== NUEVA SECCIÓN DE ACCESIBILIDAD POR AUDIO =====
        st.markdown("### 🎤 **Accesibilidad por Audio**")
        
        # Opción 1: Leer respuestas automáticamente
        auto_leer = st.checkbox("🔊 Leer respuestas automáticamente", 
                               value=st.session_state.auto_leer,
                               help="El bot leerá en voz alta sus respuestas (requiere gtts)")
        st.session_state.auto_leer = auto_leer
        
        # Opción 2: Entrada por voz
        st.markdown("#### 🎙️ Entrada por Voz")
        st.caption("Alternativa a escribir para personas con dificultades")
        
        if not STT_AVAILABLE:
            st.warning("⚠️ Instala SpeechRecognition para entrada por voz:\n`pip install SpeechRecognition`")
        
        # Usar st.audio_input para grabar audio
        voz_input = st.audio_input("Grabar pregunta", key="voz_input_sidebar")
        
        if voz_input and not st.session_state.procesando_voz:
            st.session_state.procesando_voz = True
            with st.spinner("🔄 Procesando tu voz..."):
                texto_voz, error = procesar_entrada_voz(voz_input)
                
                if texto_voz:
                    st.success(f"✅ Dijiste: \"{texto_voz}\"")
                    # Guardar como si fuera un prompt escrito
                    st.session_state.voz_prompt = texto_voz
                else:
                    st.error(f"❌ {error}")
            st.session_state.procesando_voz = False
        
        st.markdown("---")

        # Control de velocidad
        st.markdown("### ⚙️ Modo de velocidad")
        modo_rapido = st.checkbox("🚀 Modo rápido (respuestas más ágiles)", 
                                  value=st.session_state.modo_rapido,
                                  help="Activa para respuestas más rápidas con procesamiento limitado")
        st.session_state.modo_rapido = modo_rapido
        
        if modo_rapido:
            st.info("⚡ Modo rápido activado: procesamiento optimizado")
        else:
            st.info("🐢 Modo completo: máxima precisión (puede ser más lento)")

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
                st.session_state.audio_states = {}
                st.session_state.audio_cache = {}
                st.rerun()

        st.markdown("---")

        # Análisis de documentos
        st.markdown("### 📎 Análisis de Documentos")
        st.markdown("**Formatos soportados:**")
        st.markdown("📊 Excel, CSV, JSON, XML")
        st.markdown("📄 PDF, Word, TXT")
        st.markdown("🖼️ Imágenes (OCR)")
        st.markdown("💻 Detección automática de código")
        
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
                key="usar_doc_en_analisis_check",
            )

            if st.button("🔍 Procesar documento"):
                with st.spinner("📄 Procesando..."):
                    texto_doc, datos_numericos = extraer_texto_de_archivo(uploaded_file)
                    
                    if texto_doc and len(texto_doc.strip()) > 0:
                        vectorstore_doc = crear_vectorstore_desde_texto(texto_doc)
                        if vectorstore_doc:
                            st.session_state.doc_vectorstore = vectorstore_doc
                            st.session_state.doc_nombre = uploaded_file.name
                            st.session_state.datos_numericos = datos_numericos
                            st.session_state.usar_doc_en_analisis = True
                            st.success("✅ Documento listo para consultas!")
                            
                            # Mostrar resumen
                            if datos_numericos:
                                with st.expander("Ver resumen del análisis"):
                                    st.markdown(datos_numericos.generar_resumen_estadistico())
                    else:
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
- 💻 Análisis de Código Fuente
- 💬 Memoria de Conversación
- 🎧 **Texto a Voz (accesibilidad)**
- 🎤 **Entrada por Voz (accesibilidad)**
- ⚡ Streaming Avanzado
            """
        )

        st.markdown("---")
        st.markdown("### 📊 Estado del Sistema")
        st.success(f"✅ Modelo: {MODEL_NAME}")
        st.info(f"📁 Memoria: {len(st.session_state.messages)} mensajes")
        if st.session_state.modo_rapido:
            st.info("⚡ Modo: Rápido")
        else:
            st.success("🐢 Modo: Completo")
        
        # Estado de accesibilidad
        if TTS_AVAILABLE:
            st.success("🔊 TTS: Disponible")
        else:
            st.warning("🔇 TTS: No instalado")
        
        if STT_AVAILABLE:
            st.success("🎤 STT: Disponible")
        else:
            st.warning("🎙️ STT: No instalado")
            
        if st.session_state.auto_leer:
            st.info("🔊 Auto-lectura: Activada")
            
        if st.session_state.datos_numericos:
            if st.session_state.datos_numericos.dataframes:
                st.info(f"📊 DataFrames: {len(st.session_state.datos_numericos.dataframes)}")
            if st.session_state.datos_numericos.numeros_extraidos:
                st.info(f"🔢 Valores numéricos: {len(st.session_state.datos_numericos.numeros_extraidos)}")
            if st.session_state.datos_numericos.codigo_detectado:
                st.info(f"💻 Código detectado: Sí")
        st.markdown("---")
        st.caption("CC-SA: 2026 AguweyBot PRO - Versión Accesible")

    # Contenido principal
    st.markdown(
        '<h1 style="color: #00ffff; text-shadow: 0 0 20px rgba(0,255,255,0.5);">🎧 AguweyBot PRO Accesible</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Sistema cognitivo con memoria, RAG, análisis de documentos y funcionalidades de audio")

    # Cargar retriever base
    with st.spinner("🚀 Inicializando sistemas cognitivos..."):
        retriever_base = cargar_retriever()
        if retriever_base is None and os.path.exists(KNOWLEDGE_FILE):
            st.warning(
                "⚠️ No se pudo inicializar la base vectorial. Se responderá solo con el modelo base (sin RAG)."
            )

    # Mostrar historial
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(
                    f'<div class="respuesta-aguwey">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Añadir controles de audio para mensajes anteriores
                msg_id = f"msg_{i}"
                crear_controles_audio(message["content"], msg_id)
            else:
                st.markdown(message["content"])

    # Verificar si hay un prompt desde voz
    prompt_voz = st.session_state.get("voz_prompt", None)
    if prompt_voz:
        prompt = prompt_voz
        # Limpiar el prompt de voz para no procesarlo de nuevo
        st.session_state.voz_prompt = None
    else:
        # Entrada normal de texto
        prompt = st.chat_input("Escribe tu consulta (o usa el micrófono en la barra lateral)...")

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
            st.info(f"📚 Usando documento: {st.session_state.doc_nombre}")
            retriever_actual = st.session_state.doc_vectorstore.as_retriever(
                search_kwargs={"k": 20}
            )
        elif retriever_base:
            st.info("📚 Usando conocimiento base")
            retriever_actual = retriever_base

        # Buscar documentos
        with st.spinner("🔍 Buscando en el documento..."):
            if retriever_actual:
                try:
                    docs = retriever_actual.invoke(prompt)
                    if docs and len(docs) > 0:
                        st.success(f"✅ Encontrados {len(docs)} fragmentos relevantes")
                        
                        with st.expander("Ver fragmentos encontrados"):
                            for i, doc in enumerate(docs[:3]):
                                st.text(f"Fragmento {i+1}: {doc.page_content[:200]}...")
                    else:
                        st.warning("⚠️ No se encontraron fragmentos relevantes")
                        
                        st.info("🔄 Intentando búsqueda más general...")
                        docs = retriever_actual.invoke("información contenido documento")
                        if docs:
                            st.success(f"✅ Búsqueda general encontró {len(docs)} fragmentos")
                except Exception as e:
                    st.error(f"Error en búsqueda: {str(e)}")
                    docs = []
            else:
                st.warning("⚠️ No hay retriever disponible")
                docs = []

        # Construir mensajes
        mensajes = construir_mensajes_con_historial(
            prompt, 
            docs if docs else None,
            st.session_state.datos_numericos if st.session_state.usar_doc_en_analisis else None
        )

        # Generar respuesta
        with st.chat_message("assistant"):
            respuesta = mostrar_respuesta_streaming(mensajes)
            
            if respuesta:
                st.session_state.messages.append(
                    {"role": "assistant", "content": respuesta}
                )
                
                msg_id = f"msg_{len(st.session_state.messages) - 1}"
                
                if st.session_state.auto_leer and TTS_AVAILABLE:
                    if msg_id not in st.session_state.audio_cache:
                        audio_html, audio_base64 = texto_a_voz(respuesta)
                        if audio_base64:
                            st.session_state.audio_cache[msg_id] = audio_base64
                            
                            if msg_id not in st.session_state.audio_states:
                                st.session_state.audio_states[msg_id] = {
                                    "mostrar_reproductor": True,
                                    "audio_generado": True
                                }
                            
                            st.markdown(f"""
                            <div style="margin-top: 10px; padding: 10px; background: rgba(0,255,255,0.1); border-radius: 10px;">
                                <div style="display: flex; justify-content: flex-end; margin-bottom: 5px;">
                                    <button onclick="this.parentElement.parentElement.style.display='none'" 
                                            style="background: transparent; border: 1px solid #00ffff; color: #00ffff; border-radius: 15px; padding: 2px 10px; cursor: pointer;">
                                        ✖ Cerrar
                                    </button>
                                </div>
                                <audio controls autoplay style="width: 100%; border-radius: 30px;">
                                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                </audio>
                            </div>
                            """, unsafe_allow_html=True)
                
                crear_controles_audio(respuesta, msg_id)

        # Mostrar fuentes consultadas
        if docs and not st.session_state.modo_rapido:
            with st.expander("📚 Fuentes consultadas"):
                if st.session_state.doc_nombre:
                    st.markdown(f"**Documento:** {st.session_state.doc_nombre}")
                for i, doc in enumerate(docs[:3], 1):
                    st.markdown(f"**Fragmento {i}:**")
                    st.caption(doc.page_content[:300] + "...")

    # Mensaje de bienvenida inicial
    elif st.session_state.first_interaction:
        st.info(
            "👋 **¡Bienvenido a la versión accesible!** \n\n"
            "Puedes interactuar de dos formas:\n"
            "✍️ **Escribiendo** tus preguntas\n"
            "🎤 **Usando el micrófono** en la barra lateral (si tienes SpeechRecognition instalado)\n\n"
            "También puedes:\n"
            "- Subir documentos (PDF, Word, Excel, CSV, JSON, XML, imágenes)\n"
            "- Activar **lectura automática** de respuestas en la barra lateral\n"
            "- Usar el botón **🔊 Escuchar** junto a cada respuesta\n\n"
            "**¡NUEVO!** Los botones de audio ahora son persistentes - no desaparecen al hacer clic."
        )
        st.session_state.first_interaction = False

    # Footer
    st.markdown(
        """
        <div class="fixed-footer">
        <strong>⚡ Licencia CC-SA</strong> • Prof. Raymond Rosa Ávila • AguweyBot PRO 2026 • <strong>🎧 Versión Accesible</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()