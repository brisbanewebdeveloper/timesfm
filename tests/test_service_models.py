import pytest

from timesfm.service_models import ForecastRequest


def test_forecast_request_assigns_default_series_names() -> None:
  request = ForecastRequest(inputs=[[1.0, 2.0], [3.0, 4.0]], horizon=4)

  assert request.series_names_or_default() == ["series_1", "series_2"]


@pytest.mark.parametrize(
  ("inputs", "message"),
  [
    ([[1.0, float("inf")]], "finite numeric values"),
    ([[]], "must not be empty"),
  ],
)
def test_forecast_request_rejects_invalid_inputs(
  inputs: list[list[float]],
  message: str,
) -> None:
  with pytest.raises(ValueError, match=message):
    ForecastRequest(inputs=inputs, horizon=2)


def test_forecast_request_requires_aligned_series_names() -> None:
  with pytest.raises(ValueError, match="same length as inputs"):
    ForecastRequest(
      inputs=[[1.0, 2.0], [3.0, 4.0]],
      horizon=2,
      series_names=["only-one"],
    )
