"""External integrations (weather, exchange rates, LLM providers)."""

import logging

# httpx logs full request URLs at INFO; the ExchangeRate key lives in the URL path and the
# OpenWeather key in the query string, so keep that logger quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
