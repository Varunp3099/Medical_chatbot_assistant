import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ===================== CONFIG =====================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"
UPLOAD_DIR = "./uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===================== EMBEDDING MODEL =====================
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===================== PINECONE SETUP =====================
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # MiniLM embedding size
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

# ===================== MAIN FUNCTION =====================
def load_vectorstore(uploaded_files):
    file_paths = []

    # Save uploaded PDFs
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # Process each PDF
    for file_path in file_paths:
        print(f"📄 Loading PDF: {file_path}")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        print(f"🔍 Creating embeddings for {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)

        print("📤 Uploading vectors to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting") as progress:
            index.upsert(
                vectors=list(zip(ids, embeddings, metadatas))
            )
            progress.update(len(embeddings))

        print(f"✅ Completed indexing for {file_path}")
