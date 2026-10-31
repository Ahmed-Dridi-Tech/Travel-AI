"""Curated travel FAQ. Answers are deliberately general and tell the user to verify rules that
change (visas, entry requirements); the assistant relays them, it never improvises."""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class FaqEntry:
    """One FAQ topic with multilingual trigger keywords (matched on accent-folded text)."""

    topic: str
    keywords: tuple[str, ...]
    answer: str


FAQ: tuple[FaqEntry, ...] = (
    FaqEntry("visa", ("visa", "viza", "فيزا", "تاشيرة"),
        "Les règles de visa dépendent de votre nationalité et de votre destination. Pour entrer en Tunisie, "
        "de nombreux pays (dont ceux de l'Union européenne) bénéficient d'une entrée sans visa pour un séjour "
        "touristique, généralement jusqu'à 90 jours. Pour voyager depuis la Tunisie, la plupart des pays "
        "(espace Schengen, Royaume-Uni, etc.) exigent un visa à demander auprès du consulat ou de son centre "
        "agréé. Vérifiez toujours auprès de l'ambassade ou du consulat avant de réserver."),
    FaqEntry("documents", ("document", "papier", "passeport", "passport", "باسبور", "وثائق"),
        "Documents usuels : passeport en cours de validité (souvent au moins 3 à 6 mois après la date de retour "
        "selon la destination), visa si requis, billet retour ou de continuation, justificatif d'hébergement et "
        "assurance voyage (obligatoire pour un visa Schengen). Les exigences exactes varient selon la "
        "destination : vérifiez auprès du consulat."),
    FaqEntry("best_time", ("meilleure periode", "quand partir", "best time", "when to visit", "saison", "best season", "وقتاش", "wa9tech"),
        "En Tunisie, le printemps (avril-juin) et l'automne (septembre-novembre) sont les périodes les plus "
        "agréables pour la plupart des visites. L'été (juin-septembre) convient à la plage mais il fait très "
        "chaud, surtout dans le sud. L'hiver est idéal pour le Sahara (Douz, Tozeur), avec des nuits fraîches."),
    FaqEntry("currency", ("monnaie", "carte bancaire", "especes", "liquide", "payer par carte", "pay by card", "عملة"),
        "La monnaie est le dinar tunisien (TND). Il est en principe interdit d'exporter des dinars : changez ou "
        "dépensez vos dinars avant de partir et conservez vos reçus de change. Les cartes sont acceptées dans les "
        "hôtels et grands commerces ; prévoyez du liquide pour les souks, taxis et petits restaurants."),
    FaqEntry("vaccination", ("vaccin", "vaccine", "vaccination", "تلقيح", "لقاح"),
        "Aucun vaccin n'est généralement exigé pour entrer en Tunisie, sauf un certificat de fièvre jaune pour "
        "les voyageurs venant de pays à risque. Consultez votre médecin ou un centre de vaccination avant le "
        "départ pour vérifier que vos vaccins de routine sont à jour."),
    FaqEntry("tipping", ("pourboire", "tipping", "bakchich", "بقشيش"),
        "Le pourboire est d'usage mais non obligatoire : environ 5 à 10 % au restaurant si le service n'est pas "
        "inclus, et une petite pièce pour les porteurs, guides et chauffeurs."),
    FaqEntry("emergency", ("urgence", "emergency", "police", "ambulance", "samu", "pompier", "طوارئ", "اسعاف"),
        "Numéros d'urgence en Tunisie : Police 197, SAMU (urgences médicales) 190, Protection civile "
        "(pompiers) 198, Garde nationale 193."),
    FaqEntry("language", ("langue", "language", "parle", "speak", "لغة"),
        "L'arabe est la langue officielle, avec le dialecte tunisien dans la vie quotidienne. Le français est très "
        "répandu (commerce, administration, tourisme) et l'anglais est compris dans les zones touristiques."),
    FaqEntry("safety", ("securite", "safety", "danger", "arnaque", "scam", "امان"),
        "Conseils de sécurité : gardez une copie de vos documents, utilisez les taxis agréés (avec compteur) ou "
        "une application de VTC, convenez des prix avant d'acheter dans les souks, buvez de l'eau en bouteille, "
        "et consultez les recommandations officielles de votre pays avant de vous rendre en zone frontalière "
        "ou désertique."),
    FaqEntry("transport", ("transport", "louage", "taxi", "train", "metro", "bus", "se deplacer", "getting around", "تاكسي"),
        "Transports : taxis jaunes avec compteur, louages (taxis collectifs interurbains à prix fixe), trains "
        "(SNCFT) entre les grandes villes, métro léger à Tunis et vols intérieurs (par exemple vers Djerba). "
        "Les VTC sont disponibles dans les grandes villes."),
    FaqEntry("electricity", ("prise", "adaptateur", "voltage", "electric", "plug", "230v", "كهرباء"),
        "Le courant est de 230 V / 50 Hz avec des prises de type C et E (comme en France)."),
    FaqEntry("sim_internet", ("sim", "internet", "wifi", "roaming", "ooredoo", "orange", "telecom"),
        "Les opérateurs mobiles sont Ooredoo, Orange et Tunisie Telecom ; une carte SIM prépayée s'achète en "
        "boutique sur présentation du passeport."),
    FaqEntry("dress_culture", ("tenue", "habill", "dress", "ramadan", "coutume", "custom", "religion", "alcool", "alcohol"),
        "La Tunisie est un pays musulman modéré : une tenue décontractée est acceptée sur les plages et dans les "
        "hôtels, mais une tenue plus couvrante est recommandée pour les lieux religieux et les villages. "
        "Pendant le Ramadan, les horaires et l'ouverture des restaurants peuvent changer."),
    FaqEntry("baggage", ("bagage", "baggage", "luggage", "valise", "cabine", "cabin"),
        "Les franchises de bagages dépendent de la compagnie et du tarif : consultez les conditions de votre "
        "billet ou contactez directement la compagnie aérienne."),
    FaqEntry("budget", ("budget", "combien coute", "cost of", "prix moyen", "ميزانية"),
        "Le budget dépend de la destination, de la saison et du niveau de confort. Indiquez-moi la destination : "
        "notre catalogue fournit un budget indicatif pour 3 nuits à deux personnes."),
)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c)).replace("ة", "ه")


def _contains(message: str, keyword: str) -> bool:
    kw = _fold(keyword)
    tail = r"(?!\w)" if len(kw) <= 4 else ""  # short keywords must match a whole word
    return re.search(rf"(?<!\w){re.escape(kw)}{tail}", message) is not None


def find_faq(message: str) -> FaqEntry | None:
    """Best-matching FAQ entry for a user message (most/longest keyword hits), or None."""
    folded = _fold(message)
    best: tuple[int, FaqEntry] | None = None
    for entry in FAQ:
        score = sum(len(kw) for kw in entry.keywords if _contains(folded, kw))
        if score and (best is None or score > best[0]):
            best = (score, entry)
    return best[1] if best else None


def list_topics() -> list[str]:
    """Topic names, for the 'not found' answer."""
    return [e.topic for e in FAQ]
