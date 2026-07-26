from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    external_chat_id: str | None = None
    external_user_id: str | None = None
    document_id: int | None = None
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ResetRequest(BaseModel):
    conversation_id: str | None = None
    external_chat_id: str | None = None
    external_user_id: str | None = None
