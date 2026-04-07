import asyncio

import httpx
import pytest

from timesfm import mcp_server
from timesfm.service_models import ForecastResponse


def test_create_mcp_server_defaults_to_bind_host_without_localhost_only_security() -> None:
  server = mcp_server._create_mcp_server(host="0.0.0.0")

  assert server.settings.host == "0.0.0.0"
  assert server.settings.transport_security is None


def test_create_mcp_server_keeps_localhost_transport_protection() -> None:
  server = mcp_server._create_mcp_server(host="127.0.0.1")

  assert server.settings.host == "127.0.0.1"
  assert server.settings.transport_security is not None
  assert server.settings.transport_security.enable_dns_rebinding_protection is True
  assert "127.0.0.1:*" in server.settings.transport_security.allowed_hosts


def _runtime_with_transport(
  monkeypatch: pytest.MonkeyPatch,
  transport: httpx.BaseTransport,
) -> httpx.AsyncClient:
  client = httpx.AsyncClient(
    transport=transport,
    base_url="http://backend",
  )
  monkeypatch.setattr(
    mcp_server,
    "_proxy_runtime",
    mcp_server.ProxyRuntime(
      settings=mcp_server.ProxySettings(api_base_url="http://backend"),
      client=client,
    ),
  )
  return client


def test_forecast_tool_proxies_success(monkeypatch: pytest.MonkeyPatch) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert request.url.path == "/forecast"
    assert request.read() == (
      b'{"inputs":[[1.0,2.0,3.0]],"horizon":2,"series_names":["demo"]}'
    )
    return httpx.Response(
      200,
      json={
        "model_repo": "google/timesfm-2.5-200m-pytorch",
        "horizon": 2,
        "results": [
          {
            "name": "demo",
            "forecast": [4.0, 5.0],
            "lower_90": [3.0, 4.0],
            "lower_80": [3.5, 4.5],
            "median": [4.0, 5.0],
            "upper_80": [4.5, 5.5],
            "upper_90": [5.0, 6.0],
          }
        ],
      },
    )

  async def scenario() -> None:
    client = _runtime_with_transport(monkeypatch, httpx.MockTransport(handler))
    try:
      result = await mcp_server.forecast(
        inputs=[[1.0, 2.0, 3.0]],
        horizon=2,
        series_names=["demo"],
      )
    finally:
      monkeypatch.setattr(mcp_server, "_proxy_runtime", None)
      await client.aclose()

    assert isinstance(result, ForecastResponse)
    assert result.results[0].forecast == [4.0, 5.0]

  asyncio.run(scenario())


def test_forecast_tool_surfaces_backend_validation_errors(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(400, json={"detail": "Requested horizon exceeds max."})

  async def scenario() -> None:
    client = _runtime_with_transport(monkeypatch, httpx.MockTransport(handler))
    try:
      with pytest.raises(ValueError, match="Requested horizon exceeds max"):
        await mcp_server.forecast(inputs=[[1.0, 2.0, 3.0]], horizon=2)
    finally:
      monkeypatch.setattr(mcp_server, "_proxy_runtime", None)
      await client.aclose()

  asyncio.run(scenario())


def test_forecast_tool_reports_backend_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("boom", request=request)

  async def scenario() -> None:
    client = _runtime_with_transport(monkeypatch, httpx.MockTransport(handler))
    try:
      with pytest.raises(RuntimeError, match="HTTP API is unavailable"):
        await mcp_server.forecast(inputs=[[1.0, 2.0, 3.0]], horizon=2)
    finally:
      monkeypatch.setattr(mcp_server, "_proxy_runtime", None)
      await client.aclose()

  asyncio.run(scenario())


def test_fetch_backend_health_reports_degraded_status(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("boom", request=request)

  async def scenario() -> None:
    client = _runtime_with_transport(monkeypatch, httpx.MockTransport(handler))
    try:
      result = await mcp_server.fetch_backend_health()
    finally:
      monkeypatch.setattr(mcp_server, "_proxy_runtime", None)
      await client.aclose()

    assert result.status == "degraded"
    assert result.backend is None
    assert "HTTP API is unavailable" in (result.detail or "")

  asyncio.run(scenario())
