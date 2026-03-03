Licencia: Código de registro: 2603034756097; CC-NC-SA; Prof. Raymond Rosa Ávila

# ⚡ AguweyBot PRO

**Asistente técnico inteligente con RAG + streaming + interfaz profesional**  
Especializado en ingeniería, ciencias aplicadas y problemas técnicos reales.

## ✨ Características principales

- **RAG semántico** con base de conocimiento técnica (Chroma + Ollama embeddings)
- **Streaming de respuestas** en tiempo real con efecto cursor (estilo profesional)
- Interfaz **Streamlit** oscura de alto contraste optimizada para entornos técnicos
- Modelo local ligero: **phi3:mini** (rápido y eficiente en hardware modesto)
- Embeddings: **nomic-embed-text**
- Soporte para contexto largo (**4096 tokens**)
- Diseño responsivo + tema técnico inspirado en GitHub Dark / VS Code
- Sidebar con estado del sistema y visualización de documentos recuperados
- Respuestas estructuradas, rigurosas y fundamentadas en contexto recuperado

## 🛠️ Requisitos

- Python 3.10+
- Ollama instalado y corriendo localmente
- Modelos descargados:

  ```bash
  ollama pull phi3:mini
  ollama pull nomic-embed-text

~8 GB RAM recomendados (funciona en 6 GB con algo de paciencia)

🚀 Instalación rápida
Bash# 1. Clonar el repositorio
git clone https://github.com/TU-USUARIO/aguweybot-pro.git
cd aguweybot-pro

# 2. Crear y activar entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Crear/cargar tu base vectorial
#    → colocar documentos en la carpeta que usarás
#    → ejecutar tu script de ingesta (no incluido en este repo base)

# 5. Ejecutar la aplicación
streamlit run AguweyBot.py
requirements.txt recomendado
txtstreamlit>=1.31.0
langchain-community>=0.2
langchain-core>=0.2
chromadb>=0.5
ollama>=0.3
📂 Estructura recomendada del proyecto
textaguweybot-pro/
├── AguweyBot.py           # aplicación principal
├── vector_db/             # base Chroma (git ignore)
├── fondos/                # fondo.png, otras imágenes
├── logo.png               # logo personalizado (opcional)
├── docs/
│   └── screenshots/       # capturas para el README
├── .gitignore
├── README.md
└── requirements.txt
⚙️ Personalización rápida
Python# En AguweyBot.py
MODEL_NAME      = "phi3:mini"          # → puedes cambiar a "llama3.1:8b", "mistral", etc.
EMBED_MODEL     = "nomic-embed-text"   # → "mxbai-embed-large", "all-minilm", etc.
PERSIST_DIRECTORY = "vector_db"        # carpeta de la base vectorial
