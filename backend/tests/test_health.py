import asyncio

from httpx import ASGITransport, AsyncClient

from app.main import create_app


async def _get_health() -> tuple[int, dict[str, str]]:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/health")
    return response.status_code, response.json()


def test_health_reports_liveness_without_configuration() -> None:
    status_code, payload = asyncio.run(_get_health())

    assert status_code == 200
    assert payload == {"status": "ok", "api_version": "0.1.0"}
