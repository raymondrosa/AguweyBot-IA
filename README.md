Licencia: Código de registro: 2603064782073; CC-NC-SA; Prof. Raymond Rosa Ávila

📘 README.md — AguweyBot PRO
Asistente Inteligente con RAG, análisis de documentos y OCR integrado
Desarrollado por: Prof. Raymond Rosa Ávila

⚡ AguweyBot PRO
AguweyBot PRO es un asistente cognitivo avanzado para:

Ingeniería civil y ciencias aplicadas
Escritura creativa y producción literaria
Análisis profundo de documentos
Clasificación de información y recuperación semántica (RAG)
Memoria conversacional extendida
OCR para lectura de imágenes (Tesseract)

Funciona completamente de manera local, sin necesidad de internet, utilizando:

Ollama + Modelo Phi-3 ("phi3:mini")
Streamlit
ChromaDB
LangChain

Ideal para:

Docentes
Investigadores
Escritores
Estudiantes avanzados
Profesionales en ingeniería


🟦 1. Requisitos del sistema
Sistema operativo:
✔️ Windows 10 / Windows 11 (recomendado)
Software necesario:

Python 3.10 – 3.12
Ollama para Windows
Git (opcional)


⚙️ 2. Instalación paso a paso (Windows)

✔️ 2.1 Instalar Ollama

Visite: https://ollama.com/download
Descargue la versión para Windows
Instale normalmente
En CMD verifique:

ollama --version

Luego descargue el modelo:
ollama pull phi3:mini


✔️ 2.2 Descargar AguweyBot PRO
Desde GitHub:
git clone https://github.com/TU-USUARIO/AguweyBotPRO.git
cd AguweyBotPRO

O descargue el ZIP desde GitHub.

✔️ 2.3 Crear entorno virtual (recomendado)
python -m venv venv

Activarlo:
venv\Scripts\activate


✔️ 2.4 Instalar dependencias
Con el entorno activado:
pip install -r requirements.txt

Si no usa requirements.txt:
pip install streamlit langchain-community chromadb pypdf python-docx pillow pytesseract openpyxl pandas


🔤 3. Instalación de Tesseract OCR (Opcional pero recomendado)
Permite a AguweyBot leer imágenes con texto.
✔️ Descargar Tesseract para Windows
Página oficial:
https://github.com/UB-Mannheim/tesseract/wiki
Descargue:
👉 tesseract-ocr-w64-setup-5.x.x.exe
✔️ Durante la instalación:

Mantenga la ruta por defecto:

C:\Program Files\Tesseract-OCR\


MARQUE esta opción importante:

✔️ “Add Tesseract to the system PATH”
✔️ Verifique la instalación:
En CMD escriba:
tesseract --version

Debe mostrar la versión.

▶️ 4. Ejecutar AguweyBot PRO
Con el entorno virtual activado:
streamlit run AguweyBot.py

La aplicación abrirá en:
http://localhost:8501


📂 5. Estructura del proyecto
AguweyBotPRO/
│── AguweyBot.py
│── conocimiento.txt
│── vector_db/              # Generado automáticamente
│── fondo.png               # Opcional
│── logo.png                # Opcional
│── venv/                   # Entorno virtual
│── requirements.txt


📎 6. Funcionalidades principales
🔬 Modo Técnico (Ingeniería)

Explicaciones rigurosas
Estructuras, suelos, carreteras, BIM, geotecnia
Solución de problemas paso a paso

✍️ Modo Creativo (Literatura)

Análisis narrativo
Redacción creativa
Corrección de estilo
Desarrollo de personajes y tramas

📚 RAG Avanzado

Recuperación semántica con ChromaDB
Puede usar:

conocimiento.txt
documentos cargados por el usuario



📄 Análisis automático de documentos
Formatos soportados:

PDF
Word (.docx)
TXT
Excel
Imágenes (con OCR)

📸 OCR (opcional)

Lee texto en imágenes
Perfecto para fotos de pizarras, notas o escaneos

⚡ Streaming

Respuestas en tiempo real al estilo ChatGPT


🧠 7. Flujo interno del sistema
Usuario
  ↓
Sube documento / escribe consulta
  ↓
Extracción de texto (PDF/Word/Excel/OCR)
  ↓
Segmentación en fragmentos (chunking)
  ↓
Vectorización → Nomic Embed
  ↓
Búsqueda semántica en ChromaDB (RAG)
  ↓
Construcción dinámica del prompt
  ↓
Modelo Phi-3 (Ollama)
  ↓
Respuesta en Streamlit (con streaming)


🛠️ 8. Errores comunes y soluciones
❗ "tesseract no se reconoce"

Agregar al PATH:

C:\Program Files\Tesseract-OCR\

❗ No lee PDF
Instalar:
pip install pypdf

❗ No lee docx
Instalar:
pip install python-docx

❗ Excel falla
Instalar:
pip install openpyxl

❗ Ollama no responde
Reiniciar Windows después de instalar Ollama.

🧑‍🏫 9. Créditos
Desarrollado por:
Prof. Raymond Rosa Ávila
AguweyBot PRO — 2026
