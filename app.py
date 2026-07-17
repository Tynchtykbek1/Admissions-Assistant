from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from document_processor import extract_text_from_txt

app = FastAPI(title="Admissions RAG Assistant")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "Admissions RAG Assistant is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    content = await file.read()

    with open(file_path, "wb") as saved_file:
        saved_file.write(content)

    extracted_text = None

    if file_path.suffix == ".txt":
        extracted_text = extract_text_from_txt(file_path)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "saved_to": str(file_path),
        "extracted_text": extracted_text
    }