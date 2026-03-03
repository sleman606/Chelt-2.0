# Value Finder v1

Deterministic, production-minded UK horse racing value finder.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -e .[dev]
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and update API endpoints if needed.

Set The Racing API credentials (HTTP Basic Auth):

```bash
setx RACING_API_USERNAME "your_username"
setx RACING_API_PASSWORD "your_password"
```

If you prefer header token auth, set `api.auth_mode: "apikey"` and:

```bash
setx RACING_API_KEY "your_key"
```

## CLI

```bash
python -m value_finder run --date YYYY-MM-DD --config config.yaml
python -m value_finder run --date YYYY-MM-DD --config config.yaml --refresh-minutes 30
python -m value_finder calibrate --from YYYY-MM-DD --to YYYY-MM-DD --config config.yaml
python -m value_finder ingest-results --date YYYY-MM-DD --config config.yaml
```

## Cron (06:00 UK)

```cron
0 6 * * * TZ=Europe/London python -m value_finder run --date $(date +\%F) --config /path/to/config.yaml
```

## Notes

- API endpoints are centralized in config and may need adjustment for your Racing API schema.
- Runs are deterministic from input data + config.
- Missing data is fail-soft and annotated in notes.
