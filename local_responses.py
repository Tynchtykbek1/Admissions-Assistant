import re


LOCAL_RESPONSES = {
    "identity": {
        "ru": "Я Admissions Assistant — бот-помощник по вопросам поступления.",
        "en": "I’m Admissions Assistant, a bot that helps with admissions questions.",
    },
    "capabilities": {
        "ru": (
            "Я отвечаю на вопросы о поступлении, необходимых документах, сроках "
            "подачи, визе, апостиле, стипендиях, контактах и других требованиях, "
            "если информация есть в базе знаний."
        ),
        "en": (
            "I answer questions about admissions, required documents, application "
            "deadlines, visas, apostilles, scholarships, contacts, and related "
            "requirements when the information is available in the knowledge base."
        ),
    },
    "manager": {
        "ru": (
            "По вопросам поступления вы можете связаться с менеджерами:\n\n"
            "• Адахан — @TheLuckiestPersonEver\n"
            "• Максат — @maksatuniguide\n\n"
            "Напишите им в Telegram, если в базе знаний нет ответа на ваш вопрос "
            "или вам нужна консультация."
        ),
        "en": (
            "For admissions assistance, you can contact the managers:\n\n"
            "• Adakhan — @TheLuckiestPersonEver\n"
            "• Maksat — @maksatuniguide\n\n"
            "You can message them on Telegram if the knowledge base does not "
            "contain the answer or if you need personal assistance."
        ),
    },
    "out_of_scope": {
        "ru": (
            "Я специализируюсь на вопросах о поступлении, документах, сроках, визе "
            "и требованиях. Задайте вопрос по этой теме."
        ),
        "en": (
            "I specialize in admissions, documents, deadlines, visas, and related "
            "requirements. Please ask a question on that topic."
        ),
    },
}

_INTENT_PHRASES = {
    "identity": {
        "ru": {
            "Как тебя зовут?",
            "Кто ты?",
            "Ты кто?",
            "Как называется этот бот?",
            "Представься",
            "Что ты за бот?",
        },
        "en": {
            "What is your name?",
            "What's your name?",
            "Who are you?",
            "What bot are you?",
            "Introduce yourself",
        },
    },
    "capabilities": {
        "ru": {
            "Что ты умеешь?",
            "На какие вопросы ты можешь ответить?",
            "Какие вопросы можно задавать?",
            "Чем ты можешь помочь?",
            "В чем ты можешь помочь?",
            "О чем тебя можно спрашивать?",
            "Что можно у тебя спросить?",
        },
        "en": {
            "What can you do?",
            "What questions can I ask?",
            "How can you help?",
            "What can I ask you?",
            "What topics do you cover?",
        },
    },
    "manager": {
        "ru": {
            "Кто твой менеджер?",
            "Кто менеджер?",
            "Кто может помочь с поступлением?",
            "Как связаться с менеджером?",
            "Как связаться с человеком?",
            "Можно поговорить с менеджером?",
            "Кому написать по поводу поступления?",
            "Кто знает все о поступлении?",
            "Кто может проконсультировать?",
            "Дай контакты менеджера",
            "С кем можно связаться?",
            "Кому обратиться за помощью?",
        },
        "en": {
            "Who is your manager?",
            "How can I contact a manager?",
            "Can I speak to a human?",
            "Who can help me with admission?",
            "Who should I contact about admissions?",
            "Can I contact an admissions manager?",
            "Who can give me personal assistance?",
            "Give me the manager contacts",
        },
    },
    "out_of_scope": {
        "ru": {
            "Какая сегодня погода?",
            "Расскажи анекдот",
            "Сколько будет 2+2?",
            "Напиши стих",
            "Кто выиграл матч?",
            "Как приготовить пиццу?",
        },
        "en": {
            "What is the weather?",
            "Tell me a joke",
            "What is 2+2?",
            "Write a poem",
            "Who won the match?",
            "How do I cook pizza?",
        },
    },
}


def normalize_local_text(text: str) -> str:
    """Normalize case, punctuation, and whitespace for conservative exact matching."""
    without_punctuation = re.sub(r"[^\w\s]", " ", text.casefold(), flags=re.UNICODE)
    return " ".join(without_punctuation.split())


_NORMALIZED_INTENT_PHRASES = {
    intent: {
        language: frozenset(normalize_local_text(phrase) for phrase in phrases)
        for language, phrases in localized_phrases.items()
    }
    for intent, localized_phrases in _INTENT_PHRASES.items()
}


def resolve_local_response(text: str, language: str) -> str | None:
    """Return an exact deterministic local response, or defer to the RAG backend."""
    if language not in {"ru", "en"}:
        return None
    normalized = normalize_local_text(text)
    if not normalized:
        return None
    for intent in ("identity", "capabilities", "manager", "out_of_scope"):
        if normalized in _NORMALIZED_INTENT_PHRASES[intent][language]:
            return LOCAL_RESPONSES[intent][language]
    return None
