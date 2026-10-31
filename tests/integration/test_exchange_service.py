"""Live ExchangeRate-API calls. Skipped when EXCHANGE_RATE_API_KEY is not set."""

import pytest

from app.core.config import get_settings
from app.integrations.exchange import ExchangeService

KEY = get_settings().exchange_rate_api_key
pytestmark = pytest.mark.skipif(not KEY, reason="EXCHANGE_RATE_API_KEY not set")


async def test_real_rate_eur_tnd() -> None:
    svc = ExchangeService(KEY)
    try:
        r = await svc.get_rate("EUR", "TND")
    finally:
        await svc.close()
    assert 3.0 < r.rate < 4.0 and r.last_updated.year >= 2026


async def test_real_common_rates() -> None:
    svc = ExchangeService(KEY)
    try:
        rates = await svc.get_common_rates()
    finally:
        await svc.close()
    assert [r.from_currency for r in rates] == ["EUR", "USD", "GBP", "CHF"] and all(r.rate > 0 for r in rates)
