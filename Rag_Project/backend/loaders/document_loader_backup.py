from pathlib import Path
from bs4 import BeautifulSoup
import fitz
import json
import re
import shutil
from langchain_chroma import Chroma
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
import pickle
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from scipy.spatial import distance
from rank_bm25 import BM25Okapi

BM25_DIR = Path("./bm25_index")
BM25_DIR.mkdir(exist_ok=True)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
UPLOAD_DIR = Path("data/uploads")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="rag_documents",
    persist_directory="./chroma_db",
)

app = FastAPI()



def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_document(path:Path):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text(path)
    elif suffix == ".md":
        return load_markdown(path)
    elif suffix == ".html":
        return load_html(path)
    elif suffix == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )
def load_text(path: Path):
    text = path.read_text(encoding="utf-8")
    return [{
        text: clean_text(text),
        "metadata":{
            "source": str(path),
            "section": None,
            "page": None,

        }
    }]

def load_markdown(path: Path):
    text = path.read_text(encoding="uft-8")
    current_heading = None
    current_text = []
    document = []
    for line in text.splitlines():
        if line.startswith("#"):
            if current_text:
                document.append({
                    "text": clean_text("".join(current_text)),
                    "metadata":{
                        "source": str(path),
                        "section": current_heading,
                        "page": None,
                    }
                })
                current_text = []
                current_heading = line.lstrip("#").strip()
        else:
            current_text.append(line + "\n")
        if current_text:
            document.append({
                "text": clean_text("".join(current_text)),
                    "metadata":{
                        "source": str(path),
                        "section": current_heading,
                        "page": None,
                    }
            })    
    return document  
    

def load_html(path: Path):
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    document = []
    for tag in soup(["script","style"]):
        tag.decompose()
    current_head = None
    current_text = []
    body = soup.body or soup
    for element in body.findall(["h1","h2","h3","h4","h5","h6","p"]):
        if element.name.startswith("h"):
            if current_text:
                document.append({
                    "text": clean_text("".join(current_text)),
                    "metadata":{
                        "source": str(path),
                        "section": current_head,
                        "page": None,
                    }
                })
                current_text = []
                current_head = element.get_text(" ", strip=True)
        else:
            current_text.append(element.get_text(" ", strip=True))
    document.append({
                    "text": clean_text("".join(current_text)),
                    "metadata":{
                        "source": str(path),
                        "section": current_head,
                        "page": None,
                    }
                })
    return document

def load_pdf(path: Path):
    doc = fitz.open(path)
    document = []
    current_heading = None
    for page_num, page in enumerate(doc):
        text = page.get_text()
        document.append({
            "page_content": clean_text(text),
            "metadata":{
                "source": str(path),
                "section": None,
                "page": page_num + 1,
            }
        })

    return document

def ingest_document(uploaded_path: str):
    # print(f"ingesting document:{uploaded_path}")
    path = Path(uploaded_path)
    raw_path = RAW_DIR / path.name

    shutil.copy(uploaded_path, raw_path)
    
    document = load_document(raw_path)
    # print(document, "docs")
    processed_path = PROCESSED_DIR / f"{path.stem}.json"

    with open(processed_path, "w", encoding="utf-8") as f:
         json.dump(
            document,
            f,
            indent=2,
            ensure_ascii=False
        )
    return document

# if __name__ == "__main__":
#     ingest_document("data/raw/sample.pdf")
