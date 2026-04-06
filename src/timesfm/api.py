"""FastAPI service wrapper for TimesFM inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from .service_runtime import (
  RuntimeSettings,
  format_series_forecast,
  forecast_arrays,
  load_torch_model,
)
from .service_models import (
  ForecastRequest,
  ForecastResponse,
  HealthResponse,
  SeriesForecast,
)


def _env_int(name: str, default: int) -> int:
  """Reads an integer environment variable with a fallback."""
  raw_value = os.getenv(name)
  if raw_value is None:
    return default
  try:
    return int(raw_value)
  except ValueError as exc:
    raise RuntimeError(f"Environment variable {name} must be an integer.") from exc


def _load_settings() -> RuntimeSettings:
  """Loads service runtime settings from environment variables."""
  return RuntimeSettings(
    model_repo=os.getenv("TIMESFM_MODEL_REPO", RuntimeSettings.model_repo),
    max_context=_env_int("TIMESFM_MAX_CONTEXT", RuntimeSettings.max_context),
    max_horizon=_env_int("TIMESFM_MAX_HORIZON", RuntimeSettings.max_horizon),
    per_core_batch_size=_env_int(
      "TIMESFM_PER_CORE_BATCH_SIZE", RuntimeSettings.per_core_batch_size
    ),
  )


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Loads the TimesFM model once for the service lifetime."""
  settings = _load_settings()
  model = load_torch_model(settings)
  app.state.runtime_settings = settings
  app.state.model = model
  yield


app = FastAPI(title="TimesFM Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
  """Reports service status and CUDA visibility."""
  import torch

  settings: RuntimeSettings = app.state.runtime_settings
  return HealthResponse(
    status="ok",
    model_repo=settings.model_repo,
    max_context=settings.max_context,
    max_horizon=settings.max_horizon,
    per_core_batch_size=settings.per_core_batch_size,
    cuda_built=torch.backends.cuda.is_built(),
    cuda_available=torch.cuda.is_available(),
    cuda_device_count=torch.cuda.device_count(),
  )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
  """Runs a forecast request against the preloaded model."""
  import torch

  settings: RuntimeSettings = app.state.runtime_settings
  if request.horizon > settings.max_horizon:
    raise HTTPException(
      status_code=400,
      detail=(
        "Requested horizon exceeds the configured max horizon. "
        f"{request.horizon} > {settings.max_horizon}."
      ),
    )

  try:
    point_forecast, quantile_forecast = forecast_arrays(
      app.state.model,
      request.inputs,
      request.horizon,
    )
  except torch.OutOfMemoryError as exc:
    raise HTTPException(
      status_code=503,
      detail=(
        "Forecast request exceeded available GPU memory. Reduce the request size "
        "or lower TIMESFM_MAX_CONTEXT / TIMESFM_PER_CORE_BATCH_SIZE in the service "
        "configuration."
      ),
    ) from exc
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc

  series_names = request.series_names_or_default()
  results = []
  for index, name in enumerate(series_names):
    results.append(
      SeriesForecast(name=name, **format_series_forecast(point_forecast[index], quantile_forecast[index]))
    )

  return ForecastResponse(
    model_repo=settings.model_repo,
    horizon=request.horizon,
    results=results,
  )


def main() -> None:
  """Runs the TimesFM REST API with Uvicorn."""
  uvicorn.run(
    "timesfm.api:app",
    host=os.getenv("TIMESFM_HOST", "0.0.0.0"),
    port=_env_int("TIMESFM_PORT", 8000),
  )
