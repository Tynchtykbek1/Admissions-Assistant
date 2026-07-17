from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from document_processor import extract_text_from_txt, split_text_into_chunks

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
    chunks = []

    if file_path.suffix == ".txt":
        extracted_text = extract_text_from_txt(file_path)
        chunks = split_text_into_chunks(extracted_text)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "saved_to": str(file_path),
        "text_length": len(extracted_text) if extracted_text else 0,
        "chunks_count": len(chunks),
        "first_chunk": chunks[0] if chunks else None
    }