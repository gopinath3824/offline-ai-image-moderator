# Offline AI Image Moderator 🛡️

A privacy-first, 100% local desktop application for Windows that analyzes images for potentially unsafe or inappropriate content using on-device AI inference.

---

## 📑 Table of Contents

1. [Overview & Privacy Guarantee](#1-overview--privacy-guarantee)
2. [Key Features](#2-key-features)
3. [Technology Stack](#3-technology-stack)
4. [Supported Moderation Categories](#4-supported-moderation-categories)
5. [Model Architecture & Limitations](#5-model-architecture--limitations)
6. [Windows Installation Guide](#6-windows-installation-guide)
7. [First-Run Setup & Model Download](#7-first-run-setup--model-download)
8. [Offline Verification Checklist](#8-offline-verification-checklist)
9. [Thresholds & Verdict Classification](#9-thresholds--verdict-classification)
10. [Packaging as Windows Standalone Executable](#10-packaging-as-windows-standalone-executable)
11. [Safety, Ethics & False Positive Handling](#11-safety-ethics--false-positive-handling)
12. [Troubleshooting & Common Errors](#12-troubleshooting--common-errors)

---

## 1. Overview & Privacy Guarantee

**Offline AI Image Moderator** is built with zero-compromise privacy as its core principle.

- **Zero Cloud Calls:** All computer vision neural inference executes locally on your CPU or GPU.
- **Zero Telemetry or Analytics:** No usage tracking, no remote logging, no external heartbeats.
- **Read-Only Image Processing:** Selected images are never modified, overwritten, re-encoded, or deleted on disk.
- **Strict Offline Enforcement:** Neural models are initialized with `local_files_only=True`, accompanied by `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` environment locks.
- **Private Local Storage:** Scan history is persisted exclusively in a local SQLite database (`~/.offline_image_moderator/history.db`). Copies of original images are never retained.

> **Note on Internet Access:** Internet connectivity is required **only once** during the initial model download step. Once model weights are saved in your local models folder, the application operates 100% offline.

---

## 2. Key Features

- **Dashboard:** At-a-glance moderation metrics (Total Scanned, Flagged, Needs Review, Likely Safe), active model status, and recent scan feed.
- **Single Image Analysis:** Interactive workspace with high-resolution image preview, EXIF orientation correction, transparency normalization, category meters, and verdict summaries.
- **Batch Moderation:** Process multiple images or full folders with a queue manager, live progress bar, pause/stop controls, error resilience, retry functionality, and CSV export.
- **Scan History:** SQLite-backed audit trail with search, status filtering, detailed score breakdown, CSV export, and record deletion.
- **Configurable Settings:** Adjustable flag and review thresholds, category toggles, model selector, CPU/CUDA preference, and file size/resolution boundaries.
- **Offline Model Manager:** In-app status monitor showing disk consumption, installed models, one-click local downloads, and uninstallation to free disk space.
- **Built-in Diagnostic Engine:** Zero-download diagnostic test engine allowing instant offline evaluation and UI verification.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ / 3.14 | Core language runtime |
| **Desktop GUI** | CustomTkinter & Tkinter | Sleek, dark-themed responsive desktop interface |
| **Image Processing** | Pillow (PIL) | EXIF transpose, format validation, transparency compositing |
| **AI Inference** | PyTorch & Hugging Face Transformers | Local neural network execution (Vision Transformers, CLIP) |
| **Storage** | SQLite | Local audit history and statistics |
| **Distribution** | PyInstaller | Standalone Windows `.exe` packaging |

---

## 4. Supported Moderation Categories

| Category | Display Name | Scope & Visual Signals |
| :--- | :--- | :--- |
| `nsfw` | NSFW Content | General explicit or adult content detected by baseline models. |
| `nudity` | Nudity & Sexually Explicit | Exposed genitalia, sexual acts, or anatomical nudity. |
| `suggestive` | Suggestive / Erotic | Provocative poses, revealing lingerie, or sexualized imagery. |
| `violence` | Violence & Physical Harm | Aggressive physical assaults, weapons in active violence, trauma. |
| `gore` | Gore & Disturbing Imagery | Severe bodily injury, mutilation, blood, and open wounds. |
| `hate_symbols` | Hate Symbols & Extremism | Recognized hate group insignia and extremist symbols. |

---

## 5. Model Architecture & Limitations

The application features a modular classifier system:

### 1. Falconsai NSFW Classifier (`falconsai_nsfw`) — Baseline
- **Architecture:** Vision Transformer (`google/vit-base-patch16-224` fine-tuned).
- **Scope:** Binary classification (`normal` vs `nsfw`).
- **Disk Size:** ~340 MB.
- **License:** Apache 2.0.
- **Important Limitation:** This model only provides adult/NSFW discrimination. It **does not** detect violence, gore, or hate symbols.

### 2. CLIP Multi-Category Zero-Shot Engine (`clip_zero_shot`)
- **Architecture:** Contrastive Language-Image Pre-Training (`openai/clip-vit-base-patch32`).
- **Scope:** Multi-category detection across Nudity, Suggestive, Violence, Gore, and Hate Symbols.
- **Disk Size:** ~605 MB.
- **License:** MIT.
- **Operation:** Evaluates visual feature embeddings against localized zero-shot text descriptors completely on-device.

### 3. Offline Diagnostic Engine (`synthetic_diagnostic`) — Built-In Test Mode
- **Scope:** Built-in heuristic engine assessing color histograms, skin-tone chromaticity, red-channel trauma density, and deterministic cryptographic hashes.
- **Disk Size:** 0 MB (built-in).
- **Purpose:** Instant testing and CI validation without downloading multi-hundred-megabyte weights.

---

## 6. Windows Installation Guide

### Step 1: Install Python
Ensure Python 3.11 or later is installed. During Python installation, check the box:
> ☑ **Add python.exe to PATH**

Verify in PowerShell or Command Prompt:
```powershell
python --version
```

### Step 2: Clone or Download the Repository
Navigate to your desired project folder:
```powershell
cd "c:\Users\acer\Documents\imersive 1"
```

### Step 3: Create & Activate a Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (Or in Command Prompt: .\venv\Scripts\activate.bat)
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 7. First-Run Setup & Model Download

### Option A: Via the Desktop GUI (Recommended)
1. Launch the application:
   ```powershell
   python main.py
   ```
2. Navigate to **⚙️ Settings & Models** in the left sidebar.
3. Scroll down to the **Offline Model Manager** section.
4. Click **⬇ Download Model** next to *Falconsai NSFW Classifier* or *CLIP Multi-Category Zero-Shot Engine*.
5. Wait for the download and file verification to complete.
6. The model is now cached locally at `~/.offline_image_moderator/models/`. You can disconnect from the internet.

### Option B: Via Command Line
Download models directly from PowerShell:
```powershell
# Download baseline Falconsai model
python main.py --download-model falconsai_nsfw

# Check model statuses
python main.py --list-models
```

### Option C: Instant Diagnostic Mode (No Internet Needed)
If you are on an air-gapped PC or want to test immediately:
1. Open **⚙️ Settings**.
2. Select **synthetic_diagnostic: Offline Diagnostic Engine** as the Active Model.
3. Click **💾 Save Settings**. You can now scan images without downloading weights.

---

## 8. REST API Server Mode 🚀

You can also run Offline AI Image Moderator as a high-performance **REST API service** using FastAPI and Uvicorn.

### Start the API Server
```powershell
# Method 1: Using server runner
py server.py --host 127.0.0.1 --port 8000

# Method 2: Via main.py flag
py main.py --api --port 8000
```

### Interactive Documentation & Endpoints
- **Swagger Interactive UI:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`
- **Health Check:** `GET http://localhost:8000/health`

### Key Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/moderate` | Moderate single image (`multipart/form-data`) |
| `POST` | `/api/v1/moderate/batch` | Moderate multiple images in batch |
| `GET` | `/api/v1/models` | List catalog models and offline readiness |
| `POST` | `/api/v1/models/download` | Trigger local model weights download |
| `GET` | `/api/v1/history` | Query scan audit trail with filters and pagination |
| `GET` | `/api/v1/history/stats` | Retrieve aggregate moderation KPI statistics |
| `GET` | `/api/v1/history/{id}` | Inspect full scores and record details |
| `DELETE`| `/api/v1/history/{id}` | Delete a scan record |
| `DELETE`| `/api/v1/history` | Clear entire audit database |

### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/api/v1/moderate" \
  -F "file=@sample_test.png" \
  -F "flag_threshold=0.70"
```

---

## 8. Offline Verification Checklist

To prove and certify that image analysis runs strictly offline without network activity:

1. **Disconnect Internet:** Unplug your Ethernet cable or disable Wi-Fi (Airplane mode).
2. **Verify Environment Variables:** Confirm in log files (`~/.offline_image_moderator/logs/app.log`) that:
   ```
   Enforced strict offline environment: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
   ```
3. **Run Single Image & Batch Analysis:** Select and analyze images. Confirm analysis completes normally.
4. **Network Monitoring Audit (Optional):**
   - Open **Windows Resource Monitor** (`resmon.exe`) -> **Network** tab.
   - Filter by `python.exe`.
   - Observe zero outgoing TCP/UDP connections and 0 bytes/sec network transfer during inference.
5. **Run Automated Offline Suite:**
   ```powershell
   pytest tests/test_batch_and_offline.py -v
   ```

---

## 9. Thresholds & Verdict Classification

Images receive one of three primary moderation verdicts:

```
[0.00 ....................... Review Threshold ....................... Flag Threshold ....................... 1.00]
        LIKELY SAFE                     NEEDS REVIEW (Uncertainty)                  FLAGGED UNSAFE
```

- **Likely Safe:** All enabled category scores remain strictly below the Review Threshold (default: `< 40%`).
- **Needs Review:** One or more category scores fall into the uncertainty band `[Review Threshold, Flag Threshold)` (default: `40% - 70%`). Requires human inspection.
- **Flagged Unsafe:** One or more enabled category scores meet or exceed the Flag Threshold (default: `≥ 70%`).

Thresholds can be adjusted dynamically in the **Settings** screen using the visual sliders.

---

## 10. Packaging as Windows Standalone Executable

To distribute the application to users who do not have Python installed:

### Step 1: Run the Automated Build Script
```powershell
python build_exe.py
```

### Step 2: Output Location
The compilation produces a standalone distribution directory at:
```
dist/OfflineAIImageModerator/OfflineAIImageModerator.exe
```

### Source vs. Packaged Executable
- **Python Source:** Requires Python and installed virtual environment libraries. Highly customizable for development.
- **Packaged Executable:** Self-contained folder containing the bundled Python interpreter, Tkinter/CustomTkinter runtimes, PyTorch binaries, and libraries. Does not require Python on the target machine. Model weights remain stored in `~/.offline_image_moderator/models/`.

---

## 11. Safety, Ethics & False Positive Handling

Automated moderation models provide recommendations, not definitive legal or moral judgments.

### Responsible Use Guidelines:
1. **Never Automatically Delete User Files:** Flagged images should be isolated for human review, never automatically purged.
2. **Artistic & Medical Context:** Classical paintings, statues, anatomical illustrations, surgical photography, and breastfeeding imagery frequently generate high nudity scores. Always retain human manual review for ambiguous cases.
3. **Skin Exposure Heuristics:** High skin tone exposure (e.g. at a beach, athletic events) is not proof of sexual content.
4. **No Identity or Facial Recognition:** This tool does not perform facial recognition, identity tracing, or biometric profiling.

---

## 12. Troubleshooting & Common Errors

### `ModelNotInstalledError: Model weights not found at ...`
- **Cause:** The selected neural model has not been downloaded to your local drive yet.
- **Fix:** Go to **Settings -> Offline Model Manager** and click **Download Model**, or switch to **synthetic_diagnostic** for offline evaluation.

### `CUDA out of memory` or `Torch not compiled with CUDA`
- **Cause:** Your system does not have an NVIDIA GPU or sufficient VRAM.
- **Fix:** In **Settings**, change **Inference Device Preference** to `cpu`. The application runs on CPU inference.

### `Unsupported format '.xyz'` or `Corrupted image`
- **Cause:** Only `.jpg`, `.jpeg`, `.png`, `.webp`, and `.bmp` files are supported.
- **Fix:** Convert the image to PNG/JPEG or verify the file is not 0 bytes or corrupted.

### Running Pytest Test Suite
To verify all application components:
```powershell
pytest -v tests/
```
All 25 automated tests will execute and validate database operations, scoring rules, image loaders, validation checks, and offline isolation.

---

## 📄 License

This desktop application is released under the **MIT License**. Third-party model weights are subject to their respective upstream licenses (Falconsai: Apache 2.0; OpenAI CLIP: MIT).
