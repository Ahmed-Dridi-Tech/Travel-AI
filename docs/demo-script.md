# Démo Rihla — Script 20 minutes

> Les six messages ci-dessous ont été vérifiés de bout en bout (langue, intention, outil, réponse) en mode
> « léger » (Railway, NLP par mots-clés) **et** en mode complet (NLP local) : résultats identiques.

## Avant la réunion (10 min)

- [ ] Ouvrir l'application Streamlit (local : `streamlit run frontend/streamlit_app.py`, ou l'URL Railway).
- [ ] Vérifier `/health` → `"db":"connected"`.
- [ ] Envoyer **1 message test**, puis effacer la conversation (🗑️). *Le message test crée une conversation :
      faites-le **avant** de préparer les données ci-dessous.*
- [ ] Choisir l'état du tableau de bord :
  - **A — histoire « en direct » (recommandé)** : `python scripts/seed_demo_data.py --reset`
    → tableau vide ; après la démo il montre exactement vos 6 conversations.
  - **B — tableau déjà vivant** : `python scripts/seed_demo_data.py`
    → 5 conversations des 3 derniers jours déjà présentes (+ vos 6 en direct = 11).
- [ ] Clé météo OpenWeather valide (sinon la réponse 1 devient « je n'ai pas pu obtenir la météo » — voir *Si ça se passe mal*).
- [ ] Clé ExchangeRate-API configurée (`EXCHANGE_RATE_API_KEY`). **Sans elle, le taux affiché est une valeur de
      démonstration (1 EUR = 3,35 TND) et la réponse le dit** : ne présentez pas ce chiffre comme réel.
- [ ] Mot de passe du tableau de bord (`DASHBOARD_PASSWORD`) à portée de main, tableau de bord dans un 2ᵉ onglet.

> **Une conversation par message.** Dans la barre latérale, cliquez **🗑️ Effacer la conversation** avant chaque
> message : chaque message devient alors sa propre conversation sur le tableau de bord. Sinon les 6 messages
> forment une seule conversation.

> **Langue des réponses.** Tant que `LLM_PROVIDER=mock`, les réponses sont des phrases-modèles (FR / EN / AR) :
> correctes mais plus « mécaniques » qu'avec un vrai modèle de langage. Avec `LLM_PROVIDER=anthropic` (crédits
> requis) elles deviennent naturelles. Les réponses de la FAQ restent en français dans les deux cas.

## Introduction (2 min)

« Je vous présente Rihla, un assistant voyage qui répond aux clients en français, anglais et dialecte tunisien
— arabe et latin. Il tourne 24 h/24, répond instantanément, et transmet les demandes de vol et d'hôtel à votre
équipe. C'est un prototype : je viens recueillir votre avis sur ce qui serait utile à une agence. »

## Démo live (10 min) — ces 6 messages, dans l'ordre

| # | Message | Ce qu'il faut montrer | Réponse attendue |
|---|---|---|---|
| 1 | `Bonjour, je voudrais savoir la météo à Djerba cette semaine` | Réponse en français, **données météo réelles** + prévisions des prochains jours | Conditions actuelles à Djerba, puis « Prévisions : jj/mm : min–max °C, … » |
| 2 | `كيفاش الطقس في سوسة ؟` | Détection automatique de l'arabe tunisien ; réponse **en arabe** | « في Sousse الطقس الآن … الحرارة …°م » (sidebar : 🇹🇳 Arabe tunisien) |
| 3 | `nheb naaref el jaw fi Tozeur` | Dialecte **latin** (« arabizi ») compris | Météo de Tozeur (sidebar : 🇹🇳 Tunisien (latin)) |
| 4 | `What's the exchange rate for euro to dinar?` | Anglais supporté ; réponse **en anglais** | « 1 EUR = x TND » (⚠️ valeur de démonstration si pas de clé ExchangeRate) |
| 5 | `Je cherche un vol Tunis Paris pour 2 personnes` | Demande **enregistrée** avec la route « Tunis → Paris » et 2 voyageurs | « J'ai enregistré votre demande de vol (réf. …). Un conseiller vous rappellera… » |
| 6 | `hôtel pas cher à Hammamet pour le weekend` | Demande d'hôtel enregistrée (budget « pas cher », week-end) | « J'ai enregistré votre demande d'hôtel (réf. …) » |

Dire à voix haute : « le robot ne devine rien : la météo vient d'un service réel, et pour les vols et hôtels
il ne promet rien — il enregistre la demande pour votre équipe. »

## Tableau de bord (5 min)

Passez à **📊 Tableau de bord** et saisissez le mot de passe. Avec l'option **A**, vous devez voir :

- **6 conversations**, **12 messages**, **2 demandes à rappeler** (vol + hôtel) ;
- **taux d'automatisation 67 %** (4 conversations sur 6 réglées sans conseiller) ;
- langues : français 3 · arabe tunisien 1 · tunisien latin 1 · anglais 1 ;
- intentions : météo 3 · change 1 · vol 1 · hôtel 1.

Phrases utiles :
- « Voici les conversations qu'on vient de faire, avec la langue détectée pour chacune. »
- « Les 2 demandes de vol et d'hôtel apparaissent ici, avec le message exact du client, pour votre équipe. »
- « Le taux d'automatisation, c'est la part des échanges réglés sans intervention humaine. »

## Si ça se passe mal

- **Météo indisponible** : le robot répond « je n'ai pas pu obtenir la météo… souhaitez-vous que j'enregistre
  une demande ? » — présentez-le comme le comportement voulu (il n'invente jamais une donnée).
- **Première réponse lente** (service Railway qui se réveille) : passez au message suivant, revenez-y après.
- **Tableau de bord « Impossible de charger »** : l'API est injoignable ; rechargez la page après vérification de `/health`.

## Questions à poser (3 min)

1. « Combien de demandes similaires recevez-vous par semaine ? »
2. « Quelle langue utilisent le plus vos clients ? »
3. « Qu'est-ce qui vous prendrait le plus de temps si l'assistant le gérait ? »
4. « Est-ce que vous utilisez WhatsApp Business pour les clients ? »
5. « Qu'est-ce qui vous manque pour faire confiance à cet outil ? »

## Ce qu'on ne montre pas

- Le code et l'architecture technique.
- Les métriques NLP (précision, F1) et les erreurs passées.

À éviter aussi : présenter l'outil comme un produit fini, ou promettre ce qui n'existe pas encore (réservation
réelle de vols/hôtels, WhatsApp, voix). Dites simplement : « prototype pour recueillir votre avis ».

## Limites connues (à connaître avant de répondre aux questions)

- Pas de recherche ni de réservation réelle de vols/hôtels : seulement l'enregistrement d'une demande.
- Descriptions détaillées disponibles pour 10 destinations sur 25 ; les 15 autres donnent région, atouts, meilleurs mois et budget indicatif.
- Le budget indicatif par destination est une estimation (3 nuits, 2 personnes), pas un prix.
- Réponses de la FAQ (visa, urgences, transports…) rédigées en français et à relire avant usage client.
- Le tableau de bord n'a qu'un mot de passe partagé : pas de comptes utilisateurs.

## Signal d'achat à écouter

- **OUI commercial** : « on pourrait l'intégrer à notre site », « combien ça coûte », « vous pouvez le faire en arabe aussi ? »
- **NON poli** : « c'est intéressant », « on va réfléchir », « envoyez-nous une doc »
