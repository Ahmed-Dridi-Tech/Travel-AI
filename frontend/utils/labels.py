"""Human-readable labels shared by the chat and dashboard pages."""

LANGUAGE_LABELS = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "ar": "🇸🇦 Arabe",
    "tun_ar": "🇹🇳 Arabe tunisien",
    "tun_lat": "🇹🇳 Tunisien (latin)",
    "unknown": "🌐 Détection...",
}

INTENT_LABELS = {
    "weather": "🌤️ Météo",
    "currency": "💱 Change",
    "destination": "🗺️ Destination",
    "flight_inquiry": "✈️ Vol",
    "hotel_inquiry": "🏨 Hôtel",
    "travel_faq": "❓ FAQ voyage",
    "smalltalk": "💬 Conversation",
    "out_of_domain": "🚫 Hors sujet",
}

INQUIRY_TYPE_LABELS = {"flight": "✈️ Vol", "hotel": "🏨 Hôtel", "package": "📦 Forfait", "other": "Autre"}
INQUIRY_STATUS_LABELS = {"pending": "⏳ À traiter", "handled": "✅ Traitée", "closed": "🔒 Clôturée"}


def language_label(code: str) -> str:
    """Label for a language code; unknown codes are shown as-is."""
    return LANGUAGE_LABELS.get(code, code)


def intent_label(code: str) -> str:
    """Label for an intent code; unknown codes are shown as-is."""
    return INTENT_LABELS.get(code, code)
