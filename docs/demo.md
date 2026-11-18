# PathSnap demo and two-laptop setup

## Before the demo

The API laptop needs the data folders and a locally trained model bundle. The UI laptop needs the repository's actual `results/experiments.csv` and `results/model_comparison.csv` for offline model comparison; `results/part_c/student_test_comparison.json` is optional for the HMM comparison. Keep the API laptop on the same trusted network as the UI laptop and note its private IPv4 address.

Expected API laptop files:

```text
PathSnap/
  train/
  student_test/
  artifacts/model_bundle.joblib
```

`configs/base.json` and model configuration should match the model bundle. This checkout currently contains 80 training trips, 10 student-test trips, and actual model-comparison outputs, but no trained bundle. The API will return a clear 503 prediction error until a model bundle is supplied.

## Laptop 1: FastAPI

On Windows PowerShell, start the API with `scripts/start_backend.ps1`:

```powershell
$env:PATHSNAP_DATA_DIR = (Get-Location).Path
$env:PATHSNAP_API_HOST = '0.0.0.0'
$env:PATHSNAP_API_PORT = '8000'
.\scripts\start_backend.ps1 -DataDir $PWD -HostAddress 0.0.0.0 -Port 8000
```

The API laptop still needs the dataset and a compatible trained model bundle.

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

On Windows PowerShell, start the UI with `scripts/start_frontend.ps1`:

```powershell
$env:PATHSNAP_API_URL = 'http://<LAPTOP_1_PRIVATE_IP>:8000'
.\scripts\start_frontend.ps1 -ApiUrl "http://<LAPTOP_1_PRIVATE_IP>:8000"
```

Find Laptop 1's private IP address (for example, with `hostname -I` on Linux), then run:

```bash
cd ~/projects/PathSnap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PATHSNAP_API_URL="http://<LAPTOP_1_PRIVATE_IP>:8000"
bash scripts/run_ui.sh
```

Open `http://localhost:8501` on Laptop 2. The sidebar should show **API connected · model loaded**. Select a reserved trip and scoring model, then choose **Reconstruct route**. Confirm that the map shows blue GPS observations and each orange predicted segment. Metrics and timings come from the API response.

If the API URL cannot connect, verify both laptops are on the same network, the IP is correct, port 8000 is allowed from Laptop 2's private IP by Laptop 1's firewall, and the API was started with host `0.0.0.0`.

## Windows PowerShell

The PowerShell scripts resolve the repository from their own script location, so they can be called while the shell is in another directory. Run them in a PowerShell window with Python 3.10+ available:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& "$HOME\projects\PathSnap\scripts\setup.ps1" -DataDir "$HOME\projects\PathSnap"
& "$HOME\projects\PathSnap\scripts\start_all.ps1" -DataDir "$HOME\projects\PathSnap"
```

For two laptops, run this on Laptop 1:

```powershell
& "$HOME\projects\PathSnap\scripts\start_backend.ps1" -DataDir "$HOME\projects\PathSnap" -HostAddress 0.0.0.0 -Port 8000
```

And this on Laptop 2, substituting Laptop 1's private IPv4 address:

```powershell
& "$HOME\projects\PathSnap\scripts\start_frontend.ps1" -ApiUrl "http://<LAPTOP_1_PRIVATE_IP>:8000"
```

If Windows Firewall prompts, allow Python on the **Private** network. To add a narrower inbound rule from an elevated PowerShell (substitute Laptop 2's private IPv4 address):

```powershell
$Laptop2PrivateIp = Read-Host 'Laptop 2 private IPv4 address'
New-NetFirewallRule -DisplayName PathSnap-API -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress $Laptop2PrivateIp -Profile Private
```

Remove that rule after the demo with `Remove-NetFirewallRule -DisplayName PathSnap-API`. Windows commands are documented but have not been executed on Windows in this environment.

On Ubuntu with UFW, restrict the API port to Laptop 2's private address:

```bash
sudo ufw allow from <LAPTOP_2_PRIVATE_IP> to any port 8000 proto tcp
```

## One-laptop fallback

On Windows PowerShell, run `scripts/start_all.ps1 -DataDir $PWD`. It starts the API, waits for health, then runs Streamlit. Press Ctrl+C in that window to stop both services.

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
4. Open **Model comparison**. Explain that the view reads the actual experiment and model-family comparison CSVs; the HMM comparison appears when C's per-trip JSON is available.
5. Mention that the UI does not request or display reference-route geometry or per-observation labels.

## Screenshots checklist

- API `/health` response with the model loaded.
- API `/trips` response listing all ten reserved trips.
- Streamlit trip selector and a completed map with observations and predicted route.
- Aggregate evaluation metrics and all three timing values.
- Model-comparison table and chart from the actual B outputs; capture the HMM comparison only if the per-trip JSON is available.
- Optional API-unavailable screenshot showing the actionable connection error.

Use only screenshots from a real run. Do not fill empty results with example numbers.
