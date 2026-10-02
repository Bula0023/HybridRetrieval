from pathlib import Path
from bs4 import BeautifulSoup
import fitz
import json
import re
import shutil

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
    text = soup.get_text()
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
    for page_num, page in enumerate(doc):
        text = page.get_text()
        document.append({
            text: clean_text(text),
            "metadata":{
                "source": str(path),
                "section": None,
                "page": page_num + 1,
            }
        })

    return document

def ingest_document(uploaded_path: str):
    path = Path(uploaded_path)
    raw_path = RAW_DIR / path.name

    shutil.copy(uploaded_path, raw_path)
    
    document = load_document(raw_path)
    processed_path = PROCESSED_DIR / f"{path.stem}.json"

    with open(processed_path, "w", encoding="utf-8") as f:
         json.dump(
            document,
            f,
            indent=2,
            ensure_ascii=False
        )

    return documents

if __name__ == "__main__":
    ingest_document("data/raw/sample.pdf")

