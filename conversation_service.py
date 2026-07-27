import logging
from dataclasses import dataclass

import app_settings
from database import (
    clear_conversation_messages,
    count_document_chunks,
    get_document,
    get_latest_document,
    get_or_create_conversation,
    synchronize_conversations_active_document,
    update_active_document,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemDocumentState:
    configured: bool
    document: dict | None
    failure_category: str | None = None


class DocumentSelectionConflict(ValueError):
    pass


class SystemDocumentUnavailable(ValueError):
    def __init__(self, category: str, conversation: dict | None = None):
        super().__init__("The system knowledge document is unavailable.")
        self.category = category
        self.conversation = conversation


def is_system_document_configured() -> bool:
    return (
        app_settings.SYSTEM_DOCUMENT_ID is not None
        or app_settings.SYSTEM_DOCUMENT_ID_INVALID
    )


def get_system_document_state() -> SystemDocumentState:
    document_id = app_settings.SYSTEM_DOCUMENT_ID
    if app_settings.SYSTEM_DOCUMENT_ID_INVALID:
        logger.error(
            "system_document_unavailable document_id=%s category=invalid_configuration",
            document_id,
        )
        return SystemDocumentState(True, None, "invalid_configuration")
    if not is_system_document_configured():
        return SystemDocumentState(False, None)

    document = get_document(document_id)
    if document is None:
        logger.error(
            "system_document_unavailable document_id=%s category=document_missing",
            document_id,
        )
        return SystemDocumentState(True, None, "document_missing")
    if count_document_chunks(document_id) < 1:
        logger.error(
            "system_document_unavailable document_id=%s category=document_empty",
            document_id,
        )
        return SystemDocumentState(True, None, "document_empty")
    return SystemDocumentState(True, document)


def synchronize_system_document_conversations() -> SystemDocumentState:
    state = get_system_document_state()
    if state.document is not None:
        synchronize_conversations_active_document(state.document["id"])
    return state


def resolve_conversation(
    *,
    conversation_id: str | None,
    external_chat_id: str | None,
    external_user_id: str | None,
    allow_latest_document_default: bool,
    requested_document_id: int | None = None,
) -> dict:
    channel = "telegram" if external_chat_id else "web"
    chat_id = external_chat_id or conversation_id or "default-local"
    state = get_system_document_state()

    if state.configured and requested_document_id is not None:
        configured_id = (
            state.document["id"]
            if state.document is not None
            else app_settings.SYSTEM_DOCUMENT_ID
        )
        if requested_document_id != configured_id:
            raise DocumentSelectionConflict(
                "The configured system document cannot be changed by a client."
            )

    latest = (
        get_latest_document()
        if not state.configured and allow_latest_document_default
        else None
    )
    default_document_id = (
        state.document["id"]
        if state.document is not None
        else latest["id"] if latest else None
    )
    conversation = get_or_create_conversation(
        channel,
        chat_id,
        external_user_id,
        conversation_id=conversation_id,
        default_document_id=default_document_id,
    )
    if state.configured:
        if state.document is None:
            raise SystemDocumentUnavailable(
                state.failure_category or "unavailable",
                conversation,
            )
        synchronize_conversations_active_document(state.document["id"])
        conversation["active_document_id"] = state.document["id"]
    elif requested_document_id is not None:
        if get_document(requested_document_id) is None:
            raise ValueError("The requested document does not exist.")
        update_active_document(conversation["id"], requested_document_id)
        conversation["active_document_id"] = requested_document_id
    return conversation


def reset_conversation(
    *,
    conversation_id: str | None,
    external_chat_id: str | None,
    external_user_id: str | None,
) -> dict:
    unavailable = False
    try:
        conversation = resolve_conversation(
            conversation_id=conversation_id,
            external_chat_id=external_chat_id,
            external_user_id=external_user_id,
            allow_latest_document_default=not external_chat_id,
        )
    except SystemDocumentUnavailable as error:
        if error.conversation is None:
            raise
        conversation = error.conversation
        unavailable = True
    cleared = clear_conversation_messages(conversation["id"])
    result = {
        "conversation_id": conversation["id"],
        "cleared_messages": cleared,
        "active_document_id": conversation["active_document_id"],
    }
    if unavailable:
        result["status"] = "system_document_unavailable"
    return result
