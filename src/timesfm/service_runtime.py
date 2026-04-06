"""Shared runtime helpers for TimesFM forecasting services and scripts."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .configs import ForecastConfig
from .timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

DEFAULT_MODEL_REPO = TimesFM_2p5_200M_torch.DEFAULT_REPO_ID

_QUANTILE_INDEXES: dict[str, int] = {
  "lower_90": 1,
  "lower_80": 2,
  "median": 5,
  "upper_80": 8,
  "upper_90": 9,
}


@dataclasses.dataclass(frozen=True)
class RuntimeSettings:
  """Reusable runtime configuration for TimesFM inference."""

  model_repo: str = DEFAULT_MODEL_REPO
  max_context: int = 1024
  max_horizon: int = 256
  per_core_batch_size: int = 32
  normalize_inputs: bool = True
  use_continuous_quantile_head: bool = True
  force_flip_invariance: bool = True
  infer_is_positive: bool = True
  fix_quantile_crossing: bool = True


def build_forecast_config(settings: RuntimeSettings) -> ForecastConfig:
  """Builds a ForecastConfig from shared runtime settings."""
  return ForecastConfig(
    max_context=settings.max_context,
    max_horizon=settings.max_horizon,
    normalize_inputs=settings.normalize_inputs,
    use_continuous_quantile_head=settings.use_continuous_quantile_head,
    force_flip_invariance=settings.force_flip_invariance,
    infer_is_positive=settings.infer_is_positive,
    fix_quantile_crossing=settings.fix_quantile_crossing,
    per_core_batch_size=settings.per_core_batch_size,
  )


def load_torch_model(settings: RuntimeSettings) -> TimesFM_2p5_200M_torch:
  """Loads and compiles the default torch TimesFM model."""
  import torch

  torch.set_float32_matmul_precision("high")

  model = TimesFM_2p5_200M_torch.from_pretrained(settings.model_repo)
  model.compile(build_forecast_config(settings))
  return model


def forecast_arrays(
  model: Any,
  inputs: list[np.ndarray] | list[list[float]],
  horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Runs a forecast for one or more numeric series."""
  compiled_config = getattr(model, "forecast_config", None)
  if compiled_config is None:
    raise RuntimeError("Model is not compiled. Please call compile() first.")
  if horizon > compiled_config.max_horizon:
    raise ValueError(
      "Requested horizon exceeds compiled max horizon. "
      f"{horizon} > {compiled_config.max_horizon}."
    )

  array_inputs = [np.asarray(series, dtype=np.float32) for series in inputs]
  return model.forecast(horizon=horizon, inputs=array_inputs)


def format_series_forecast(
  point_forecast: np.ndarray,
  quantile_forecast: np.ndarray,
) -> dict[str, list[float]]:
  """Formats one TimesFM forecast row into a JSON-friendly payload."""
  result = {"forecast": point_forecast.tolist()}
  for key, index in _QUANTILE_INDEXES.items():
    result[key] = quantile_forecast[:, index].tolist()
  return result


def format_named_forecasts(
  series_names: list[str],
  point_forecast: np.ndarray,
  quantile_forecast: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
  """Formats batched TimesFM output keyed by series name."""
  if len(series_names) != len(point_forecast):
    raise ValueError(
      "Series names and forecast outputs must have matching lengths. "
      f"{len(series_names)} != {len(point_forecast)}."
    )

  return {
    name: format_series_forecast(point_forecast[index], quantile_forecast[index])
    for index, name in enumerate(series_names)
  }
