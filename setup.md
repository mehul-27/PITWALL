# PitWall — local setup and what GitHub does *not* include

This guide is for anyone cloning the repo (e.g. a teammate) who wants to run the **web chat** and **telemetry SQL** path. Training / Groq / Kaggle are optional and noted separately.

## What you get from GitHub

- Application code (`pitwall/app/…`), UI (`static/`, `templates/`)
- Pipeline scripts (`pitwall/pipeline/…`), config, training utilities
- **Not** in the repo (by design — see `.gitignore`):
  - **`pitwall/data/pitwall.db`** — SQLite database with laps/sessions (large; build locally or copy in)
  - **`pitwall/data/cache/**`** — FastF1 on-disk cache (huge; re-downloads when you extract)
  - **`.env`** — API keys and local overrides (you create it)
  - `__pycache__/`, `.venv/`, local IDE folders
  - Most `*.md` files are ignored; **`setup.md`** is tracked as an exception

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** (3.11–3.13 common on Windows) | 64-bit recommended |
| **Git** | To clone and pull |
| **Disk space** | Several GB for Python packages; **tens of GB** if you run a full multi-season FastF1 extract + cache |
| **Ollama** (recommended for chat) | [ollama.com](https://ollama.com) — local LLM for `/chat` when `PITWALL_INFERENCE_BACKEND=ollama` |
| **Optional: NVIDIA GPU + CUDA** | Speeds local Transformers / bitsandbytes path; Ollama can use GPU depending on Ollama install |

## 1. Clone and enter the project

```powershell
git clone https://github.com/mehul-27/PITWALL.git
cd PITWALL
```

(Use your fork URL if different.)

## 2. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Python dependencies

From the **repository root** (folder that contains `pitwall/`):

```powershell
pip install -U pip
pip install -r pitwall/requirements.txt
```

This pulls **Flask, FastF1, torch, transformers,** etc. First install can take a while and needs a stable network.

- **Kaggle / training only:** see `pitwall/training/requirements_kaggle.txt` in addition, if you run those notebooks or scripts.

## 4. Environment variables (`.env`)

Create **`pitwall/.env`** (next to `pitwall/config.py`). It is **not** committed.

Minimal example for **Ollama chat** (default backend):

```env
# Optional: use if you add Groq to pipeline scripts
# GROQ_API_KEY=your_key_here

# Inference (defaults shown — omit to use these)
# PITWALL_INFERENCE_BACKEND=ollama
# OLLAMA_URL=http://127.0.0.1:11434
# OLLAMA_MODEL=llama3.2:3b
```

- **`GROQ_API_KEY`** — only needed for pipeline steps that call Groq (e.g. some dataset generation). **Not** required to open the Flask UI with Ollama.
- **Local Qwen** — advanced: set `PITWALL_INFERENCE_BACKEND=local_qwen` and paths `PITWALL_QWEN_BASE_MODEL_PATH`, `PITWALL_QWEN_ADAPTER_PATH` per `config.py` comments.

`python-dotenv` loads `pitwall/.env` automatically if present.

## 5. Ollama and the chat model

1. Install and start **Ollama** on your machine.
2. Pull a model (must match or be compatible with `OLLAMA_MODEL` in `config.py`, default `llama3.2:3b`):

   ```text
   ollama pull llama3.2:3b
   ```

3. If you use a **custom Ollama model name** (e.g. a fine-tuned `pitwall` GGUF), set `OLLAMA_MODEL` in `.env` to that name.

If Ollama is not running, the app may fall back to **stub** responses (no real model).

## 6. Database and FastF1 data (required for **telemetry** answers)

The chat can answer **general** strategy questions without a DB. For **“compare laps / Silverstone 2024 Q / sectors”** the app queries **`pitwall/data/pitwall.db`**.

That file is **not** on GitHub. You must either:

### Option A — Build from FastF1 (full control)

1. Ensure **`pitwall/data/`** exists (the pipeline creates subfolders as needed).
2. From repo root, after `pip install` and a working FastF1 install, run the extractor, e.g. one season first:

   ```powershell
   cd pitwall
   python pipeline/extract.py --season 2024
   ```

   A full multi-year run is `python pipeline/extract.py` (no `--season`) but takes much longer and uses more disk.

3. The script creates/updates `pitwall.db` and `data/cache/…` (gitignored cache).

### Option B — Copy a `pitwall.db` from a teammate

- Place the file at **`pitwall/data/pitwall.db`**
- Same relative path on every machine. Optional: they can zip `pitwall.db` only (not the whole `cache` unless you need offline repeat runs without re-downloading).

**Session-on-demand:** the app can try to **ingest a missing session** from FastF1 the first time you query it (if extract logic and network are available); you still need a working DB and usually some cached data for reliability.

## 7. Run the web app

The Flask entry point expects the working directory to be **`pitwall/`** (so imports resolve), from repo root:

```powershell
cd pitwall
python app/app.py
```

Open **http://127.0.0.1:5000** in a browser (default in code; see `app.py` if port changed).

- **API:** `POST /chat` with JSON `{"message": "…"}` (optional `session_id`).

## 8. What your friend must have *aside* from Git

| Item | Why |
|------|-----|
| **Python + venv** | Run the app and pipeline |
| **Installed deps** | `pip install -r pitwall/requirements.txt` |
| **Ollama + pulled model** | Real chat replies (unless using local Qwen or accepting stubs) |
| **`pitwall/.env`** | Keys/paths; can be empty except when using Groq or custom model paths |
| **`pitwall/data/pitwall.db`** + optionally **cache** | Telemetry SQL and lap sectors in the UI |
| **Time + disk** | First FastF1 extract and cache |
| **Network** (first time) | `pip`, Ollama pulls, FastF1 downloads for extract |

## 9. Optional: training and evaluation

- Datasets: under `pitwall/data/dataset/` (often generated, not always in repo)
- Kaggle: extra requirements in `pitwall/training/requirements_kaggle.txt`
- Groq: `GROQ_API_KEY` in `.env` when running scripts that call Groq

## 10. Troubleshooting

- **`No Data Found` / empty telemetry** — DB missing, wrong year/session, or no laps after filters. Build or copy `pitwall.db`; confirm session exists for that circuit/year (e.g. quali `Q`).
- **Stub / generic answers** — Ollama not running, wrong model name, or inference import failed; check app logs.
- **Torch / CUDA errors** — align Python and PyTorch with your GPU driver; or use Ollama only to avoid custom torch stacks.
- **Line endings** — Windows may show LF/CRLF warnings from Git; safe to ignore for this project.

## 11. Security

- Never commit **`.env`**, **API keys**, or your personal **`pitwall.db`** if it contains anything sensitive.
- Use `FLASK_SECRET_KEY` in production and turn off `debug=True` in `app.py` when exposing the server beyond localhost.

---

*Season coverage in the product is aligned with `SEASONS` in `pitwall/config.py` (e.g. 2022–2025 for the data pipeline; UI may show “2022–25” in the ticker.)*
