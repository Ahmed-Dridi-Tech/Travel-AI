"""Deterministic offline provider: formats tool data from the prompt, never calls a network.

Replies in the customer's language (French, English or Arabic) so a demo without LLM credits still
shows multilingual behaviour. It only restates what the tool data contains.
"""

import json
import re
from typing import Any

from app.integrations.llm.base import BaseLLMProvider, LLMMessage, LLMResponse

TOOL_NAME_LABEL = "TOOL_NAME:"
TOOL_DATA_LABEL = "TOOL_DATA:"
TOOL_ERROR_LABEL = "TOOL_ERROR:"
DEMO_SUFFIX = " (Valeurs de démonstration, à titre indicatif seulement.)"  # kept for callers/tests
DEMO_FLAG = "demo_data_no_api_key"

_MONTHS = {
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "ar": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
}
_TAGS = {
    "fr": {"beach": "plage", "desert": "désert", "culture": "culture", "family": "famille", "history": "histoire",
           "nature": "nature", "luxury": "luxe", "budget": "petit budget"},
    "en": {"beach": "beach", "desert": "desert", "culture": "culture", "family": "family", "history": "history",
           "nature": "nature", "luxury": "luxury", "budget": "budget-friendly"},
    "ar": {"beach": "شاطئ", "desert": "صحراء", "culture": "ثقافة", "family": "عائلات", "history": "تاريخ",
           "nature": "طبيعة", "luxury": "فخامة", "budget": "اقتصادي"},
}
_T: dict[str, dict[str, str]] = {
    "fr": {
        "ask_destination": "Pour quelle destination souhaitez-vous cette information ?",
        "weather_failed": "Je n'ai pas pu obtenir la météo pour le moment. Souhaitez-vous que j'enregistre une demande ?",
        "exchange_failed": "Je n'ai pas pu récupérer le taux de change pour le moment. Voulez-vous réessayer plus tard ?",
        "failed": "Je n'ai pas cette information pour le moment. Souhaitez-vous laisser une demande à un conseiller ?",
        "weather": "À {city}, il fait actuellement {temp}°C (ressenti {feels}°C), {desc}, humidité {hum}%, vent {wind} m/s.",
        "forecast": " Prévisions : {days}.",
        "period_note": " Ce sont les conditions actuelles ; je n'ai pas de prévision pour la période demandée.",
        "fx_conv": "{amount:g} {src} = {converted:.2f} {dst} (taux : 1 {src} = {rate:.4f} {dst}).",
        "fx_rate": "1 {src} = {rate:.4f} {dst}.",
        "suggestions": "Voici quelques destinations de notre catalogue : {names}. Laquelle vous intéresse ?",
        "dest_desc": "{name} ({region}) : {desc}",
        "dest_plain": "{name} ({region}). Points forts : {tags}.",
        "dest_months": " Meilleurs mois : {months}.",
        "dest_budget": " Budget indicatif : environ {budget} TND pour 3 nuits à deux personnes.",
        "kind_flight": "de vol", "kind_hotel": "d'hôtel",
        "recorded": "J'ai enregistré votre demande {kind} (réf. {ref}). Un conseiller vous rappellera.",
        "refine": " Pour l'affiner, pouvez-vous préciser : {field} ?",
        "need_details": "Pour votre demande {kind}, pouvez-vous préciser : {fields} ?",
        "f_destination": "la destination", "f_origin": "la ville de départ", "f_dates": "les dates", "f_travelers": "le nombre de voyageurs",
        "faq_none": "Je n'ai pas de réponse validée à cette question. Souhaitez-vous laisser une demande à un conseiller ?",
        "kw_weather": "Je peux vous donner la météo d'une destination. Laquelle vous intéresse ?",
        "kw_currency": "Je peux vérifier un taux de change. Quelle devise vous intéresse ?",
        "kw_destination": "Je peux vous renseigner sur cette destination. Que souhaitez-vous savoir ?",
        "generic": "Je suis là pour vous aider avec vos voyages en Tunisie : météo, change, destinations, vols et hôtels.",
        "demo": DEMO_SUFFIX,
    },
    "en": {
        "ask_destination": "Which destination do you mean?",
        "weather_failed": "I couldn't get the weather right now. Would you like me to record a request?",
        "exchange_failed": "I couldn't retrieve the exchange rate right now. Would you like to try again later?",
        "failed": "I don't have that information right now. Would you like to leave a request for an advisor?",
        "weather": "In {city} it is currently {temp}°C (feels like {feels}°C), {desc}, humidity {hum}%, wind {wind} m/s.",
        "forecast": " Forecast: {days}.",
        "period_note": " These are the current conditions; I have no forecast for the requested period.",
        "fx_conv": "{amount:g} {src} = {converted:.2f} {dst} (rate: 1 {src} = {rate:.4f} {dst}).",
        "fx_rate": "1 {src} = {rate:.4f} {dst}.",
        "suggestions": "Here are a few destinations from our catalogue: {names}. Which one interests you?",
        "dest_desc": "{name} ({region}): {desc}",
        "dest_plain": "{name} ({region}). Highlights: {tags}.",
        "dest_months": " Best months: {months}.",
        "dest_budget": " Indicative budget: about {budget} TND for 3 nights for two.",
        "kind_flight": "flight", "kind_hotel": "hotel",
        "recorded": "I've recorded your {kind} request (ref. {ref}). An advisor will call you back.",
        "refine": " To refine it, could you tell me {field}?",
        "need_details": "For your {kind} request, could you tell me {fields}?",
        "f_destination": "the destination", "f_origin": "the departure city", "f_dates": "the dates", "f_travelers": "the number of travellers",
        "faq_none": "I don't have a validated answer to that question. Would you like to leave it with an advisor?",
        "kw_weather": "I can give you the weather for a destination. Which one?",
        "kw_currency": "I can check an exchange rate. Which currency?",
        "kw_destination": "I can tell you about that destination. What would you like to know?",
        "generic": "I'm here to help with your trips to Tunisia: weather, exchange rates, destinations, flights and hotels.",
        "demo": " (Demo values, indicative only.)",
    },
    "ar": {
        "ask_destination": "عن أي وجهة تسأل؟",
        "weather_failed": "لم أتمكن من جلب حالة الطقس الآن. هل تريد أن أسجّل طلبًا؟",
        "exchange_failed": "لم أتمكن من جلب سعر الصرف الآن. هل تريد المحاولة لاحقًا؟",
        "failed": "ليست لدي هذه المعلومة الآن. هل تريد ترك طلب لدى مستشار؟",
        "weather": "في {city} الطقس الآن {desc}، الحرارة {temp}°م (تُحَسّ كأنها {feels}°م)، الرطوبة {hum}%، الرياح {wind} م/ث.",
        "forecast": " التوقعات: {days}.",
        "period_note": " هذه هي الأحوال الحالية؛ ليست لدي توقعات للفترة المطلوبة.",
        "fx_conv": "{amount:g} {src} = {converted:.2f} {dst} (السعر: 1 {src} = {rate:.4f} {dst}).",
        "fx_rate": "1 {src} = {rate:.4f} {dst}.",
        "suggestions": "إليك بعض الوجهات من كتالوجنا: {names}. أيّها يهمّك؟",
        "dest_desc": "{name} ({region}): {desc}",
        "dest_plain": "{name} ({region}). أبرز ما فيها: {tags}.",
        "dest_months": " أفضل الأشهر: {months}.",
        "dest_budget": " ميزانية تقريبية: حوالي {budget} دينار لـ 3 ليالٍ لشخصين.",
        "kind_flight": "رحلة طيران", "kind_hotel": "فندق",
        "recorded": "سجّلت طلبك ({kind}) (المرجع {ref}). سيتصل بك مستشار.",
        "refine": " لتحسين الطلب، هل يمكنك تحديد: {field}؟",
        "need_details": "لطلبك ({kind})، هل يمكنك تحديد: {fields}؟",
        "f_destination": "الوجهة", "f_origin": "مدينة المغادرة", "f_dates": "التواريخ", "f_travelers": "عدد المسافرين",
        "faq_none": "ليس لدي جواب مؤكَّد على هذا السؤال. هل تريد ترك سؤالك لدى مستشار؟",
        "kw_weather": "يمكنني إعطاؤك حالة الطقس في أي وجهة. أيّها تريد؟",
        "kw_currency": "يمكنني التحقق من سعر الصرف. أي عملة تريد؟",
        "kw_destination": "يمكنني إخبارك عن هذه الوجهة. ماذا تريد أن تعرف؟",
        "generic": "أنا هنا لمساعدتك في رحلاتك إلى تونس: الطقس، أسعار الصرف، الوجهات، الرحلات والفنادق.",
        "demo": " (قيم تجريبية للإرشاد فقط.)",
    },
}


def _line_value(text: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _language_of(prompt: str) -> str:
    """Reply language from the instruction line build_user_prompt puts first ('fr' when absent)."""
    first = prompt.lstrip().split("\n", 1)[0]
    if first.startswith("Respond in English"):
        return "en"
    if first.startswith("أجب"):
        return "ar"
    return "fr"


class MockLLMProvider(BaseLLMProvider):
    """Answers in French, English or Arabic and only ever restates what the tool data contains."""

    def get_model_name(self) -> str:
        return "mock-llm"

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = self._respond(last_user)
        words_in = sum(len(m.content.split()) for m in messages) + len((system_prompt or "").split())
        return LLMResponse(
            content=text,
            model=self.get_model_name(),
            input_tokens=int(words_in * 1.3),
            output_tokens=int(len(text.split()) * 1.3),
            latency_ms=50.0,
        )

    def _respond(self, prompt: str) -> str:
        lang = _language_of(prompt)
        t = _T[lang]
        tool = _line_value(prompt, TOOL_NAME_LABEL) or "none"
        error = _line_value(prompt, TOOL_ERROR_LABEL)
        raw = _line_value(prompt, TOOL_DATA_LABEL)
        data: dict[str, Any] = {}
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
        if error:
            return self._error_text(t, tool, error)
        handler = {
            "get_weather": self._weather,
            "get_exchange_rate": self._exchange,
            "get_destination_info": self._destination,
            "submit_flight_inquiry": self._inquiry,
            "submit_hotel_inquiry": self._inquiry,
            "get_faq_answer": self._faq,
        }.get(tool)
        if handler and data:
            return handler(t, lang, data) + (t["demo"] if data.get("note") == DEMO_FLAG else "")
        return self._keyword_fallback(t, prompt)

    # ------------------------------------------------------------------ formatters
    @staticmethod
    def _error_text(t: dict[str, str], tool: str, error: str) -> str:
        if error == "no_destination":
            return t["ask_destination"]
        if tool == "get_weather":
            return t["weather_failed"]
        if tool == "get_exchange_rate":
            return t["exchange_failed"]
        return t["failed"]

    @staticmethod
    def _weather(t: dict[str, str], lang: str, d: dict[str, Any]) -> str:
        city = d.get("requested_destination") or d.get("city", "")
        out = t["weather"].format(city=city, temp=f"{d['temperature_c']:.0f}", feels=f"{d['feels_like_c']:.0f}",
                                  desc=d["description"], hum=d["humidity_percent"], wind=d["wind_speed_ms"])
        if d.get("forecast"):
            days = "; ".join(
                f"{f['date'][8:10]}/{f['date'][5:7]} : {f['temp_min_c']:.0f}–{f['temp_max_c']:.0f}°C, {f['description']}"
                for f in d["forecast"]
            )
            out += t["forecast"].format(days=days)
        if d.get("period_note"):
            out += t["period_note"]
        return out

    @staticmethod
    def _exchange(t: dict[str, str], lang: str, d: dict[str, Any]) -> str:
        if d.get("converted") is not None and d.get("amount") != 1:
            return t["fx_conv"].format(amount=d["amount"], src=d["from"], dst=d["to"], converted=d["converted"], rate=d["rate"])
        return t["fx_rate"].format(src=d["from"], dst=d["to"], rate=d["rate"])

    @staticmethod
    def _destination(t: dict[str, str], lang: str, d: dict[str, Any]) -> str:
        if "suggestions" in d:
            return t["suggestions"].format(names=", ".join(s["name"] for s in d["suggestions"]))
        tags = ", ".join(_TAGS[lang].get(x, x) for x in d.get("tags", []))
        if d.get("description") and lang != "ar":  # curated descriptions exist in French and English only
            out = t["dest_desc"].format(name=d["name"], region=d["region"], desc=d["description"])
        else:
            out = t["dest_plain"].format(name=d["name"], region=d["region"], tags=tags)
        if d.get("best_months"):
            out += t["dest_months"].format(months=", ".join(_MONTHS[lang][m - 1] for m in d["best_months"]))
        if d.get("avg_budget_tnd"):
            out += t["dest_budget"].format(budget=d["avg_budget_tnd"])
        return out

    @staticmethod
    def _inquiry(t: dict[str, str], lang: str, d: dict[str, Any]) -> str:
        kind = t["kind_flight"] if d.get("inquiry_type") == "flight" else t["kind_hotel"]

        def label(field: str) -> str:
            return t.get(f"f_{field}", field)

        if d.get("status") == "need_details":
            return t["need_details"].format(kind=kind, fields=", ".join(label(f) for f in d["missing_fields"]))
        out = t["recorded"].format(kind=kind, ref=d["reference"])
        if d.get("missing_fields"):
            out += t["refine"].format(field=label(d["missing_fields"][0]))
        return out

    @staticmethod
    def _faq(t: dict[str, str], lang: str, d: dict[str, Any]) -> str:
        if d.get("found") is False:
            return t["faq_none"]
        return str(d["answer"])  # curated answers are stored in French

    @staticmethod
    def _keyword_fallback(t: dict[str, str], prompt: str) -> str:
        lowered = prompt.lower()
        if "weather" in lowered or "météo" in lowered:
            return t["kw_weather"]
        if "currency" in lowered or "taux" in lowered or "euro" in lowered:
            return t["kw_currency"]
        if "djerba" in lowered or "sousse" in lowered:
            return t["kw_destination"]
        return t["generic"]
