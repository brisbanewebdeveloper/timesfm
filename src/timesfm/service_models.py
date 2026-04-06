"""Shared request and response models for TimesFM services."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ForecastRequest(BaseModel):
  """Request payload for forecast inference."""

  model_config = ConfigDict(extra="forbid")

  inputs: list[list[float]] = Field(
    ...,
    min_length=1,
    description="One or more numeric time series contexts.",
  )
  horizon: int = Field(..., gt=0, description="Number of steps to forecast.")
  series_names: list[str] | None = Field(
    default=None,
    description="Optional names aligned to the inputs list.",
  )

  @field_validator("inputs")
  @classmethod
  def _validate_inputs(cls, value: list[list[float]]) -> list[list[float]]:
    for index, series in enumerate(value):
      if not series:
        raise ValueError(f"Input series at index {index} must not be empty.")
      if not all(math.isfinite(point) for point in series):
        raise ValueError(
          f"Input series at index {index} must contain only finite numeric values."
        )
    return value

  @model_validator(mode="after")
  def _validate_series_names(self) -> "ForecastRequest":
    if self.series_names is not None and len(self.series_names) != len(self.inputs):
      raise ValueError("series_names must have the same length as inputs.")
    return self

  def series_names_or_default(self) -> list[str]:
    """Returns caller-provided series names or deterministic defaults."""
    if self.series_names is not None:
      return self.series_names
    return [f"series_{index + 1}" for index in range(len(self.inputs))]


class SeriesForecast(BaseModel):
  """Forecast payload for one named series."""

  model_config = ConfigDict(extra="forbid")

  name: str
  forecast: list[float]
  lower_90: list[float]
  lower_80: list[float]
  median: list[float]
  upper_80: list[float]
  upper_90: list[float]


class ForecastResponse(BaseModel):
  """Response payload for batch forecast inference."""

  model_config = ConfigDict(extra="forbid")

  model_repo: str
  horizon: int
  results: list[SeriesForecast]


class HealthResponse(BaseModel):
  """Health payload for the running service."""

  model_config = ConfigDict(extra="forbid")

  status: str
  model_repo: str
  max_context: int
  max_horizon: int
  per_core_batch_size: int
  cuda_built: bool
  cuda_available: bool
  cuda_device_count: int
