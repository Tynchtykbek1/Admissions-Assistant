const fileInput = document.querySelector("#document-file");
const uploadButton = document.querySelector("#upload-button");
const uploadStatus = document.querySelector("#upload-status");
const currentDocument = document.querySelector("#current-document");
const currentFilename = document.querySelector("#current-filename");
const currentDocumentType = document.querySelector("#current-document-type");
const currentChunks = document.querySelector("#current-chunks");
const questionInput = document.querySelector("#question");
const answerMode = document.querySelector("#answer-mode");
const askButton = document.querySelector("#ask-button");
const askStatus = document.querySelector("#ask-status");
const responseSection = document.querySelector("#response-section");
const answerElement = document.querySelector("#answer");
const sourcesElement = document.querySelector("#sources");

let documentUploaded = false;
let conversationId = localStorage.getItem("admissionsConversationId");

function showError(element, message) {
    element.textContent = message;
    element.classList.add("error");
}

function clearError(element) {
    element.classList.remove("error");
}

async function readError(response) {
    try {
        const data = await response.json();
        return data.detail || "The request failed.";
    } catch {
        return "The request failed.";
    }
}

async function loadConversationStatus() {
    const query = conversationId
        ? `?conversation_id=${encodeURIComponent(conversationId)}`
        : "";
    const response = await fetch(`/conversation/status${query}`);
    if (!response.ok) {
        return;
    }
    const result = await response.json();
    conversationId = result.conversation_id;
    localStorage.setItem("admissionsConversationId", conversationId);
    documentUploaded = Boolean(result.active_document_id);
    if (result.active_document_filename) {
        currentFilename.textContent = result.active_document_filename;
        currentDocumentType.textContent = "System knowledge document";
        currentChunks.textContent = "Managed by the administrator";
        currentDocument.hidden = false;
        uploadStatus.textContent = "The system knowledge document is ready.";
    }
}

uploadButton.addEventListener("click", async () => {
    const file = fileInput.files[0];

    if (!file) {
        showError(uploadStatus, "Please select a TXT or PDF file first.");
        return;
    }

    clearError(uploadStatus);
    uploadStatus.textContent = "Uploading and processing document...";
    uploadButton.disabled = true;

    const formData = new FormData();
    formData.append("file", file);
    if (conversationId) {
        formData.append("conversation_id", conversationId);
    }

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(await readError(response));
        }

        const result = await response.json();
        conversationId = result.conversation_id;
        localStorage.setItem("admissionsConversationId", conversationId);
        const typeLabel = result.document_type === "faq" ? "FAQ" : "Standard";
        const itemCount = result.entries_count ?? result.chunks_count;
        uploadStatus.textContent = `Uploaded ${result.filename}. Text length: ${result.text_length}. Chunks / entries: ${itemCount}.`;
        currentFilename.textContent = result.filename;
        currentDocumentType.textContent = typeLabel;
        currentChunks.textContent = itemCount;
        currentDocument.hidden = false;
        await loadConversationStatus();
    } catch (error) {
        showError(uploadStatus, error.message);
    } finally {
        uploadButton.disabled = false;
    }
});

askButton.addEventListener("click", async () => {
    const question = questionInput.value.trim();

    if (!documentUploaded) {
        showError(askStatus, "Please upload a document before asking a question.");
        return;
    }

    if (!question) {
        showError(askStatus, "Please enter a question.");
        return;
    }

    const endpoint = answerMode.value === "llm" ? "/chat" : "/ask-semantic";
    clearError(askStatus);
    askStatus.textContent = "Finding an answer...";
    askButton.disabled = true;

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                question,
                conversation_id: conversationId
            })
        });

        if (!response.ok) {
            throw new Error(await readError(response));
        }

        const result = await response.json();
        if (result.conversation_id) {
            conversationId = result.conversation_id;
            localStorage.setItem("admissionsConversationId", conversationId);
        }
        answerElement.textContent = result.answer;
        sourcesElement.replaceChildren();

        if (result.sources.length === 0) {
            sourcesElement.textContent = "No relevant sources found.";
        } else {
            result.sources.forEach((source) => {
                const sourceCard = document.createElement("div");
                sourceCard.className = "source";

                const sourceTitle = document.createElement("strong");
                sourceTitle.textContent = `${source.filename} · Chunk ${source.chunk_id} · Score ${source.score.toFixed(3)}`;

                const preview = document.createElement("p");
                preview.textContent = source.preview;

                sourceCard.append(sourceTitle, preview);
                sourcesElement.append(sourceCard);
            });
        }

        responseSection.hidden = false;
        askStatus.textContent = "Answer ready.";
    } catch (error) {
        showError(askStatus, error.message);
    } finally {
        askButton.disabled = false;
    }
});

loadConversationStatus();
