from database import (
    clear_conversation_messages,
    get_latest_document,
    get_or_create_conversation,
)


def resolve_conversation(
    *,
    conversation_id: str | None,
    external_chat_id: str | None,
    external_user_id: str | None,
    allow_latest_document_default: bool,
) -> dict:
    channel = "telegram" if external_chat_id else "web"
    chat_id = external_chat_id or conversation_id or "default-local"
    latest = get_latest_document() if allow_latest_document_default else None
    return get_or_create_conversation(
        channel,
        chat_id,
        external_user_id,
        conversation_id=conversation_id,
        default_document_id=latest["id"] if latest else None,
    )


def reset_conversation(
    *,
    conversation_id: str | None,
    external_chat_id: str | None,
    external_user_id: str | None,
) -> dict:
    conversation = resolve_conversation(
        conversation_id=conversation_id,
        external_chat_id=external_chat_id,
        external_user_id=external_user_id,
        allow_latest_document_default=not external_chat_id,
    )
    cleared = clear_conversation_messages(conversation["id"])
    return {
        "conversation_id": conversation["id"],
        "cleared_messages": cleared,
        "active_document_id": conversation["active_document_id"],
    }
