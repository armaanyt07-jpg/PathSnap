# PathSnap demo and two-laptop setup

## Before the demo

The API laptop needs the data folders and a trained model bundle. The UI laptop needs the experiment CSV only if the model-comparison tab is being demonstrated. Keep the API laptop on the same trusted network as the UI laptop and note its private IPv4 address.

Expected API laptop files:

```text
PathSnap/
  train/
  student_test/
  artifacts/model_bundle.joblib
```

`configs/base.json` and model configuration should match the model bundle. This checkout currently contains the 80 training trips and 10 student-test trips, but it does not contain a trained bundle or experiment results. The API will return a clear 503 prediction error until a model bundle is supplied.

## Laptop 1: FastAPI

```bash
cd ~/projects/PathSnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PATHSNAP_DATA_DIR="$PWD"
export PATHSNAP_API_HOST=0.0.0.0
export PATHSNAP_API_PORT=8000
bash scripts/run_api.sh
```

In a second terminal on Laptop 1, verify the service and the ten reserved trips:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/trips
```

The health response should report `model_loaded: true` before making predictions. If the laptops use a firewall, allow inbound TCP port 8000 on Laptop 1 for the trusted local network. Do not expose the service to a public network.

## Laptop 2: Streamlit

Find Laptop 1's private IP address (for example, with `hostname -I` on Linux), then run:

```bash
cd ~/projects/PathSnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PATHSNAP_API_URL=http://<LAPTOP_1_PRIVATE_IP>:8000
bash scripts/run_ui.sh
```

Open `http://localhost:8501` on Laptop 2. The sidebar health indicator should show **API connected**. Select a reserved trip, choose **Reconstruct route**, then confirm that the map shows blue GPS observations and the orange predicted route. Metrics and timing values come from the API response.

If the API URL cannot connect, verify both laptops are on the same network, the IP is correct, port 8000 is allowed by Laptop 1's firewall, and the API was started with host `0.0.0.0`.

## One-laptop fallback

For local development or a backup demo, start the API and UI together:

```bash
cd ~/projects/PathSnap
export PATHSNAP_DATA_DIR="$PWD"
bash scripts/run_all.sh
```

Press Ctrl+C to stop both services.

## Demo sequence

1. Show the API health response and the ten-trip list.
2. In Streamlit, select one trip and run reconstruction.
3. Point out the GPS observations, predicted route, decoder name, aggregate metrics, and feature/inference/total timings.
4. Open **Model comparison**. Explain that the table and chart are read from the actual experiment CSV; if it is absent, the app states that no results are available.
5. Mention that the UI does not request or display reference-route geometry or per-observation labels.

## Screenshots checklist

- API `/health` response with the model loaded.
- API `/trips` response listing all ten reserved trips.
- Streamlit trip selector and a completed map with observations and predicted route.
- Aggregate evaluation metrics and all three timing values.
- Model-comparison table and chart after real experiment outputs are present.
- Optional API-unavailable screenshot showing the actionable connection error.

Use only screenshots from a real run. Do not fill empty results with example numbers.
