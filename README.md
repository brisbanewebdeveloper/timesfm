# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   All checkpoints:
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model): Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html): For your daily spreadsheet. 
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm): Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 2.5

**Archived Model Versions:**

-   1.0 and 2.0: relevant code archived in the sub directory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and [@darkpowerxo](https://github.com/darkpowerxo). 

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the support for [AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)! TimesFM [SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting) is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.


## Update - Sept. 15, 2025

TimesFM 2.5 is out!

Comparing to TimesFM 2.0, this new 2.5 model:

-   uses 200M parameters, down from 500M.
-   supports up to 16k context length, up from 2048.
-   supports continuous quantile forecast up to 1k horizon via an optional 30M
    quantile head.
-   gets rid of the `frequency` indicator.
-   has a couple of new forecasting flags.

Since the Sept. 2025 launch, the following improvements have been completed:

1.  ✅ Flax version of the model for faster inference.
2.  ✅ Covariate support via XReg (see Oct. 2025 update).
3.  ✅ Documentation, examples, and agent skill (see `timesfm-forecasting/`).
4.  ✅ Fine-tuning example with LoRA via HuggingFace Transformers + PEFT (see `timesfm-forecasting/examples/finetuning/`).
5.  ✅ Unit tests for core layers, configs, and utilities (see `tests/`).

### Install

1.  Clone the repository:
    ```shell
    git clone https://github.com/google-research/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install dependencies using `uv`:
    ```shell
    # Create a virtual environment
    uv venv

    # Activate the environment
    source .venv/bin/activate

    # Install the package in editable mode with torch
    uv pip install -e .[torch]
    # Or with the REST API framework (combine with a backend like torch to serve forecasts)
    uv pip install -e .[service]
    # Or with the MCP proxy runtime
    uv pip install -e .[mcp]
    # Or install everything needed for torch inference + both servers
    uv pip install -e .[torch,service,mcp]
    # Or with flax
    uv pip install -e .[flax]
    # Or XReg is needed
    uv pip install -e .[xreg]
    ```

3. [Optional] Install your preferred `torch` / `jax` backend based on your OS and accelerators
(CPU, GPU, TPU or Apple Silicon).:

-   [Install PyTorch](https://pytorch.org/get-started/locally/).
-   [Install Jax](https://docs.jax.dev/en/latest/installation.html#installation)
    for Flax.

### GitHub Copilot Setup

This repository already includes GitHub Copilot customization files:

-   `.github/copilot-instructions.md` for repository-wide instructions.
-   `AGENTS.md` for always-on agent guidance.
-   `timesfm-forecasting/` for the reusable TimesFM skill.

Open the repository root in VS Code so Copilot can discover those files
automatically. You do not need to copy `.github/copilot-instructions.md` or
`AGENTS.md` anywhere else.

GitHub Copilot discovers project skills from `.github/skills/` and user skills
from `~/.copilot/skills/`. To make the packaged TimesFM skill available in
Copilot agent mode, install it into one of those locations:

```shell
# Project-local
mkdir -p .github/skills
cp -r timesfm-forecasting .github/skills/

# Or user-local
mkdir -p ~/.copilot/skills
cp -r timesfm-forecasting ~/.copilot/skills/
```

After that, start a Copilot Chat session in agent mode and ask it to use the
`timesfm-forecasting` skill.

If you open a subfolder instead of the repository root, enable the VS Code
setting `chat.useCustomizationsInParentRepositories` so Copilot can still
discover the repo-level instructions and skills.

### Code Example

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```

### Local Servers

Run the existing REST API:

```shell
timesfm-api
```

Install a backend such as `.[torch,service]` before starting the REST API.

The REST API defaults to `http://127.0.0.1:8000` and still supports the same
`curl` workflow:

```shell
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/forecast \
    -H 'Content-Type: application/json' \
    -d '{
         "horizon": 12,
         "series_names": ["demo"],
         "inputs": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    }'
```

Run the MCP proxy in a second process:

```shell
TIMESFM_API_BASE_URL=http://127.0.0.1:8000 timesfm-mcp
```

The MCP endpoint is exposed at `http://127.0.0.1:8001/mcp` by default. The MCP
server wraps the existing REST API; it does not replace it.

To register that MCP proxy in VS Code, add this to `.vscode/mcp.json` in the
workspace or to your user `mcp.json`:

```json
{
    "servers": {
        "timesfm": {
            "type": "http",
            "url": "http://127.0.0.1:8001/mcp"
        }
    }
}
```

Start `timesfm-mcp` before using the server from Copilot Chat so VS Code can
discover its tools.

`AGENTS.md` and `timesfm-forecasting/SKILL.md` describe agent-skill packaging.
The MCP server is a separate runtime surface exposed by `timesfm-mcp`.

### Docker Compose Services

The repository also includes a minimal long-running REST API service and an MCP
proxy service backed by Docker Compose.

The default compose settings are conservative for shared GPUs: `max_context=512`
and `per_core_batch_size=4`. Increase them in `docker-compose.yml` if you have
more free VRAM.

The published host ports are controlled through `.env`:

```shell
TIMESFM_PORT=8000
TIMESFM_MCP_PORT=8001
```

1. Build and start both services:
    ```shell
    docker compose up --build -d
    ```

2. Check REST API health:
    ```shell
    curl http://localhost:${TIMESFM_PORT}/health
    ```

3. Send a forecast request with `curl`:
    ```shell
    curl -X POST http://localhost:${TIMESFM_PORT}/forecast \
      -H 'Content-Type: application/json' \
      -d '{
         "horizon": 12,
         "series_names": ["demo"],
         "inputs": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
      }'
    ```

4. Connect an MCP client to the proxy endpoint:
    ```shell
    echo "MCP endpoint: http://localhost:${TIMESFM_MCP_PORT}/mcp"
    ```

The compose setup keeps TimesFM inference inside the REST container. The MCP
container only forwards validated tool calls to `timesfm`, so you can keep the
existing `curl` workflow while exposing the same forecast capability to MCP
clients.

The REST container persists Hugging Face downloads in a named volume mounted at
`$HF_HOME`, so the first model download is reused across restarts.
