"""Streamable HTTP MCP server that proxies the existing TimesFM HTTP API."""

from __future__ import annotations

import dataclasses
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict

from .service_models import ForecastRequest, ForecastResponse, HealthResponse


def _env_int(name: str, default: int) -> int:
  """Reads an integer environment variable with a fallback."""
  raw_value = os.getenv(name)
  if raw_value is None:
    return default
  try:
    return int(raw_value)
  except ValueError as exc:
    raise RuntimeError(f"Environment variable {name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
  """Reads a float environment variable with a fallback."""
  raw_value = os.getenv(name)
  if raw_value is None:
    return default
  try:
    return float(raw_value)
  except ValueError as exc:
    raise RuntimeError(f"Environment variable {name} must be a float.") from exc


@dataclasses.dataclass(frozen=True)
class ProxySettings:
  """Runtime settings for the MCP proxy service."""

  api_base_url: str = "http://127.0.0.1:8000"
  request_timeout_seconds: float = 30.0
  host: str = "0.0.0.0"
  port: int = 8001


@dataclasses.dataclass(frozen=True)
class ProxyRuntime:
  """Shared client state for the MCP proxy."""

  settings: ProxySettings
  client: httpx.AsyncClient


class ProxyHealthResponse(BaseModel):
  """Health payload for the MCP proxy container."""

  model_config = ConfigDict(extra="forbid")

  status: str
  backend_url: str
  detail: str | None = None
  backend: HealthResponse | None = None


_proxy_runtime: ProxyRuntime | None = None


def _default_mcp_host() -> str:
  """Returns the MCP bind host used for transport configuration."""
  return os.getenv("TIMESFM_MCP_HOST", ProxySettings.host)


def _create_mcp_server(host: str | None = None) -> FastMCP:
  """Builds the MCP server with transport settings aligned to the bind host."""
  return FastMCP(
    "TimesFM",
    instructions=(
      "Use the forecast tool to obtain TimesFM forecasts through the existing "
      "HTTP API."
    ),
    host=host or _default_mcp_host(),
    json_response=True,
    stateless_http=True,
    streamable_http_path="/",
  )


mcp = _create_mcp_server()


def _load_proxy_settings() -> ProxySettings:
  """Loads proxy settings from environment variables."""
  api_base_url = os.getenv("TIMESFM_API_BASE_URL", ProxySettings.api_base_url)
  return ProxySettings(
    api_base_url=api_base_url.rstrip("/"),
    request_timeout_seconds=_env_float(
      "TIMESFM_API_TIMEOUT_SECONDS", ProxySettings.request_timeout_seconds
    ),
    host=os.getenv("TIMESFM_MCP_HOST", ProxySettings.host),
    port=_env_int("TIMESFM_MCP_PORT", ProxySettings.port),
  )


def _get_proxy_runtime() -> ProxyRuntime:
  """Returns the initialized proxy runtime."""
  if _proxy_runtime is None:
    raise RuntimeError("TimesFM MCP proxy runtime is not initialized.")
  return _proxy_runtime


def _extract_error_detail(response: httpx.Response) -> str:
  """Extracts a readable error payload from the backend response."""
  try:
    payload = response.json()
  except json.JSONDecodeError:
    return response.text or f"HTTP {response.status_code}"

  if isinstance(payload, dict):
    detail = payload.get("detail")
    if isinstance(detail, str):
      return detail
  return response.text or f"HTTP {response.status_code}"


async def _backend_json(
  method: str,
  path: str,
  *,
  json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Calls the backend API and returns a JSON object payload."""
  runtime = _get_proxy_runtime()
  try:
    response = await runtime.client.request(method, path, json=json_payload)
  except httpx.RequestError as exc:
    raise RuntimeError(
      "TimesFM HTTP API is unavailable. "
      f"Check TIMESFM_API_BASE_URL ({runtime.settings.api_base_url})."
    ) from exc

  if response.status_code == 400:
    raise ValueError(_extract_error_detail(response))
  if response.status_code == 503:
    raise RuntimeError(_extract_error_detail(response))
  if response.is_error:
    raise RuntimeError(
      f"TimesFM HTTP API returned {response.status_code}: "
      f"{_extract_error_detail(response)}"
    )

  payload = response.json()
  if not isinstance(payload, dict):
    raise RuntimeError("TimesFM HTTP API returned a non-object JSON payload.")
  return payload


async def run_forecast_request(request: ForecastRequest) -> ForecastResponse:
  """Forwards a validated forecast request to the REST API."""
  payload = await _backend_json(
    "POST",
    "/forecast",
    json_payload=request.model_dump(mode="json"),
  )
  return ForecastResponse.model_validate(payload)


async def fetch_backend_health() -> ProxyHealthResponse:
  """Fetches backend health for the MCP proxy."""
  runtime = _get_proxy_runtime()
  try:
    payload = await _backend_json("GET", "/health")
  except RuntimeError as exc:
    return ProxyHealthResponse(
      status="degraded",
      backend_url=runtime.settings.api_base_url,
      detail=str(exc),
    )

  return ProxyHealthResponse(
    status="ok",
    backend_url=runtime.settings.api_base_url,
    backend=HealthResponse.model_validate(payload),
  )


@mcp.tool()
async def forecast(
  inputs: list[list[float]],
  horizon: int,
  series_names: list[str] | None = None,
) -> ForecastResponse:
  """Runs a TimesFM forecast through the existing REST API."""
  request = ForecastRequest(
    inputs=inputs,
    horizon=horizon,
    series_names=series_names,
  )
  return await run_forecast_request(request)


@asynccontextmanager
async def lifespan(_app: FastAPI):
  """Initializes the HTTP client and MCP session manager."""
  global _proxy_runtime

  settings = _load_proxy_settings()
  async with AsyncExitStack() as stack:
    client = await stack.enter_async_context(
      httpx.AsyncClient(
        base_url=settings.api_base_url,
        timeout=settings.request_timeout_seconds,
      )
    )
    await stack.enter_async_context(mcp.session_manager.run())
    _proxy_runtime = ProxyRuntime(settings=settings, client=client)
    try:
      yield
    finally:
      _proxy_runtime = None


app = FastAPI(title="TimesFM MCP Proxy", version="1.0.0", lifespan=lifespan)
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/health", response_model=ProxyHealthResponse)
async def healthcheck() -> ProxyHealthResponse | JSONResponse:
  """Reports MCP proxy status and backend reachability."""
  payload = await fetch_backend_health()
  if payload.status != "ok":
    return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
  return payload


def main() -> None:
  """Runs the TimesFM MCP proxy with Uvicorn."""
  settings = _load_proxy_settings()
  uvicorn.run(
    "timesfm.mcp_server:app",
    host=settings.host,
    port=settings.port,
  )
