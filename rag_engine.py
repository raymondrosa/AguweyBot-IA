# ============================================
# RAG ENGINE - AGUWEYBOT PRO
# ============================================

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

PERSIST_DIRECTORY = "vector_db"
EMBED_MODEL = "nomic-embed-text"


def crear_base_vectorial(ruta_txt):

    if not os.path.exists(ruta_txt):
        print("Archivo conocimiento.txt no encontrado.")
        return

    loader = TextLoader(ruta_txt, encoding="utf-8")
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs_fragmentados = splitter.split_documents(documentos)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        docs_fragmentados,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    vectorstore.persist()

    print("✅ Base vectorial creada correctamente.")


if __name__ == "__main__":
    crear_base_vectorial("conocimiento.txt")