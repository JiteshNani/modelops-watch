# ModelOps Watch

Serve a model. Identify its artifact. Review distribution shifts.

An end-to-end classification and monitoring prototype using a stratified train/test split, train-only standardization, logistic regression, validated JSON model artifacts, SHA-256 fingerprints and Population Stability Index monitoring. JSON inference is tested against scikit-learn probabilities to avoid loading executable pickle artifacts.

**Status:** locally tested portfolio prototype, October 2026. Built with AI assistance; intended for hands-on study, review and extension. Not paid employment or a deployed commercial system.

## Quick start

Python 3.12 is the tested version. Dependencies are pinned to the locally tested environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the interactive demo, or /docs for API documentation.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @sample_request.json
```

Windows PowerShell alternative:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body (Get-Content sample_request.json -Raw)
```

## Architecture

```text
Stratified split → train-only scaler → logistic regression → JSON artifact → validated API → distribution review
```

- `core.py`: algorithms and validation, separate from transport.
- `app.py`: bounded Pydantic inputs, health endpoint and FastAPI interface.
- `test_core.py`: algorithm invariants, failure cases and API behavior.
- `evaluation.json`: actual saved local evaluation output; regenerate with the command above.
- `LEARNING_GUIDE.md`: walkthrough, interview questions and a practical extension exercise.
- `Dockerfile`: non-root container configuration. Docker execution has not been verified locally.
- `ci.yml`: ready-to-install CI definition; place at `.github/workflows/ci.yml` to activate.

## Measured local results

Wine dataset (UCI, included with scikit-learn): 178 samples, 133 training and 45 held-out test samples, seed 42. Saved run test accuracy 1.000 and log loss 0.0671. Perfect accuracy on 45 cases is not a production reliability claim. The drift demo intentionally shifts alcohol by +4 units and detects that feature.

## Limitations and deployment boundary

PSI is sensitive to bins, smoothing and sample size. The threshold 0.2 is illustrative, not calibrated for a business. Feature drift is not concept drift and does not directly measure accuracy loss. No automatic retraining, cloud rollout, database registry or production monitoring deployment is claimed. Model fingerprinting is an identifier, not cryptographic signing or trust verification.

Local demo only: no authentication, tenant isolation, persistence, TLS, rate limiter or deployment hardening is included. Bind to localhost for use. Request-size and field validation do not replace gateway body-size limits. Do not expose the service publicly without adding these controls. No credentials or personal data are needed.

## Design references

- [FastAPI container deployment](https://fastapi.tiangolo.com/deployment/docker/)
- [scikit-learn model persistence considerations](https://scikit-learn.org/stable/model_persistence.html)
- [scikit-learn Wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

MIT license covers this code. The Wine dataset follows the source dataset's terms; synthetic images and example policies are generated fixtures. No patient data is included.
