from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    external_chat_id: str | None = None
    external_user_id: str | None = None
    document_id: int | None = None


class ResetRequest(BaseModel):
    conversation_id: str | None = None
    external_chat_id: str | None = None
    external_user_id: str | None = None
