Licencia: Código de registro:CC-SA; Prof. Raymond Rosa Ávila

# ⚡ AguweyBot PRO

**Asistente inteligente con memoria, RAG, análisis de documentos y datos numéricos**

AguweyBot PRO es un chatbot avanzado construido con **Streamlit + Ollama + LangChain** que combina:
- Ingeniería y ciencias aplicadas
- Escritura creativa y narrativa
- Análisis automático de datos numéricos (Excel, CSV, JSON, etc.)
- RAG (Retrieval-Augmented Generation) con conocimiento base y documentos subidos
- Soporte completo para PDF, Word, Excel, CSV, JSON, XML e imágenes (OCR)

---

## ✨ Características principales

- **Memoria completa** de la conversación
- **RAG doble**: conocimiento base (`conocimiento.txt`) + documento subido
- **Análisis numérico avanzado** (media, mediana, outliers, correlaciones, gráficos automáticos)
- Soporte para **9 formatos** de documentos (incluye OCR con Tesseract)
- Streaming en tiempo real de respuestas
- Diseño moderno oscuro con efectos visuales
- Modo técnico / creativo / análisis de datos detectado automáticamente

---

## 🛠️ Requisitos

- **Python 3.10 o superior**
- **Ollama** instalado y ejecutándose
- **Tesseract OCR** (solo si usas imágenes)
- 8 GB de RAM recomendados (phi3:mini funciona bien en 8 GB)

---

## 📥 Instalación paso a paso

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/AguweyBot-PRO.git
cd AguweyBot-PRO
2. Instalar Ollama (obligatorio)

Windows / Mac / Linux: Descarga desde ollama.com
Abre una terminal y ejecuta:

Bashollama pull phi3:mini
ollama pull nomic-embed-text
3. Instalar Tesseract OCR (solo para imágenes)

Windows: Descarga desde GitHub UB-Mannheim/tesseract e instala.
macOS:Bashbrew install tesseract
Linux:Bashsudo apt update && sudo apt install tesseract-ocr tesseract-ocr-spa

4. Crear entorno virtual y instalar dependencias
Bashpython -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install streamlit langchain langchain-community langchain-core chromadb py-pdf2 python-docx pandas numpy pillow pytesseract chardet plotly matplotlib openpyxl
5. Archivos obligatorios
Crea estos archivos en la raíz del proyecto:

conocimiento.txt → (puedes dejarlo vacío o poner tu conocimiento base)
fondo.png → (imagen de fondo, opcional)
logo.png → (logo en sidebar, opcional)

Si no tienes fondo/logo, el bot sigue funcionando.

🚀 Cómo ejecutar
Bashstreamlit run AguweyBot.py
El bot se abrirá automáticamente en tu navegador.

📁 Estructura del proyecto
textAguweyBot-PRO/
├── AguweyBot.py              ← Código principal
├── conocimiento.txt          ← Base de conocimiento (RAG)
├── fondo.png                 ← Fondo personalizado
├── logo.png                  ← Logo sidebar
├── vector_db/                ← Base vectorial (se crea automáticamente)
├── README.md
└── requirements.txt          ← (puedes generar con pip freeze)

🔧 Generar requirements.txt (opcional)
Bashpip freeze > requirements.txt

🎯 Uso recomendado

Sube un documento (Excel, PDF, CSV, imagen, etc.)
Activa la opción "Usar este documento como contexto principal"
Haz preguntas técnicas, literarias o de análisis de datos
¡El bot recuerda toda la conversación!


📄 Licencia
CC-NC-SA 2026 – Prof. Raymond Rosa Ávila
Uso libre para fines educativos y personales. Prohibida la venta.
