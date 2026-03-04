Licencia: Código de registro: 2603034756097; CC-NC-SA; Prof. Raymond Rosa Ávila

📘 README.md — AguweyBot PRO
Asistente Inteligente con RAG, análisis de documentos y modos técnico/creativo

⚡ AguweyBot PRO
AguweyBot PRO es un asistente cognitivo avanzado diseñado para:

🔬 Ingeniería y ciencias aplicadas
✍️ Escritura creativa y narrativa profesional
📚 Análisis de documentos con RAG
🧠 Memoria contextual
🖼️ OCR (lectura de imágenes con texto)

Está optimizado para funcionar localmente usando:

Ollama + Modelo Phi-3 (phi3:mini)
Streamlit
RAG con ChromaDB
Lectura de PDF, Word, Excel, TXT e Imágenes


📥 1. Requisitos del sistema
🟦 Sistema operativo

Windows 10 o Windows 11

🟦 Software necesario

Python 3.10 – 3.12
Ollama (para ejecutar modelos locales)
Git (opcional para clonar el repo)


🧰 2. Instalación paso a paso

✔️ 2.1 Instalar Ollama

Entra a: https://ollama.com/download
Descarga la versión para Windows
Instálala normalmente
En CMD prueba:

ollama --version

Debe mostrar la versión instalada.

✔️ 2.2 Descargar el modelo Phi-3
Ejecuta:
ollama pull phi3:mini


✔️ 2.3 Clonar o descargar este repositorio
git clone https://github.com/TU-USUARIO/AguweyBotPRO.git

o descarga el ZIP desde GitHub.

✔️ 2.4 Crear y activar un entorno virtual (venv)
Dentro del folder del proyecto:
cd AguweyBotPRO
python -m venv venv

Activarlo:
venv\Scripts\activate

Debes ver:
(venv) C:\ruta...


✔️ 2.5 Instalar dependencias del proyecto
Con el venv activado:
pip install -r requirements.txt

Si no tienes requirements.txt, instala manualmente:
pip install streamlit langchain-community chromadb pypdf python-docx pillow pytesseract openpyxl pandas


🔤 3. Instalación de Tesseract OCR (Opcional pero recomendado)
Permite que AguweyBot lea texto en imágenes (capturas de pantalla, escaneos, pizarras, etc).

✔️ 3.1 Descargar el instalador oficial
Desde:
https://github.com/UB-Mannheim/tesseract/wiki
Descarga el instalador:
✔️ tesseract-ocr-w64-setup-5.x.x.exe
(Para Windows 64 bits)

✔️ 3.2 Instalar Tesseract
Durante la instalación:
✔️ Importante:
Marca la casilla Add Tesseract to the system PATH
Ruta por defecto recomendada:
C:\Program Files\Tesseract-OCR\


✔️ 3.3 Probar que funciona
En una nueva ventana de CMD:
tesseract --version

Debe imprimir la versión.

✔️ 3.4 Instalar puente Python-Tesseract
Con el venv activo:
pip install pytesseract


📂 4. Ejecutar AguweyBot PRO
Con Ollama corriendo en el fondo, ejecuta:
streamlit run AguweyBot.py

Automáticamente se abrirá una ventana en el navegador:
http://localhost:8501


📎 5. Funcionalidades principales
🔬 Modo Técnico (Ingeniería y ciencia)
Explicaciones rigurosas, estructuradas, con precisión conceptual.
✍️ Modo Creativo (Literatura)
Mejora de estilo, desarrollo narrativo, personajes, diálogos, etc.
📚 RAG (Retrieval Augmented Generation)
Usa un archivo conocimiento.txt o documentos cargados por el usuario.
📄 Análisis de documentos
Admite:

PDF
Word (.docx)
TXT
Excel
Imágenes (con OCR)

🧠 Memoria conversacional
Mantiene coherencia usando las últimas interacciones (MAX_HISTORY).
⚡ Streaming avanzado
Las respuestas se escriben en tiempo real.

🗂️ 6. Estructura del proyecto
AguweyBotPRO/
│
├── AguweyBot.py
├── conocimiento.txt
├── vector_db/           # generado automáticamente
├── logo.png             # opcional
├── fondo.png            # opcional
└── venv/                # entorno virtual


🔄 7. Flujo de funcionamiento interno
Usuario
   ↓
Carga documento (PDF/Word/Excel/Imagen)
   ↓
Extracción de texto
   ↓
División en fragmentos (chunking)
   ↓
Vectorización (Nomic Embed)
   ↓
Búsqueda semántica (ChromaDB)
   ↓
Construcción del prompt con contexto
   ↓
Respuesta del modelo phi3:mini vía Ollama
   ↓
Interfaz en Streamlit


🛠️ 8. Errores comunes y soluciones
❗ Tesseract no funciona en CMD
Solución: agregar al PATH:
C:\Program Files\Tesseract-OCR\

❗ No lee PDF
Asegúrate de tener:
pip install pypdf

❗ No lee Word
Instala:
pip install python-docx

❗ Excel no abre
Instala:
pip install openpyxl

❗ No encuentra Ollama
Reinicia Windows después de instalarlo.

🧑‍🏫 9. Créditos
Desarrollado por:
Prof. Raymond Rosa Ávila
AguweyBot PRO — 2026
