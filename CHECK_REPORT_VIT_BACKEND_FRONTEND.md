# Backend & Frontend Check Report – ViT Only, ResNet Removed

**Date:** Check done after switching to ViT (phase4_replay_best) only.

---

## 1. Image detection: ViT only, ResNet removed

| Item | Status |
|------|--------|
| **inference.py** | Only ViT: `ViTForImageClassification`, `phase4_replay_best.pth`. No ResNet, no `DeepfakeImageDetector`, no `GradCAM`, no `ImagePreprocessor`. |
| **Model load** | Path = `ml_models/weights/phase4_replay_best.pth` (from `MODEL_NAME` in settings). |
| **Preprocessing** | ViT: resize 224, normalize mean=0.5, std=0.5. |
| **Prediction** | `model(pixel_values=...).logits` → verdict (real/fake), confidence (0–1), probabilities. |
| **Heatmap** | ViT attention heatmap from last layer (class token → patches), overlay saved; no Grad-CAM. |

---

## 2. Backend API

| API | Method | Purpose | Status |
|-----|--------|---------|--------|
| `/api/analysis/image/` | POST | Image upload → ViT prediction + heatmap → save to DB | OK – uses `get_model_service()` → ViT only |

- **views.py**  
  - `get_model_service()` loads only ViT from `ml_models/weights/phase4_replay_best.pth`.  
  - `analyze_image()` calls `service.predict_with_heatmap()` (ViT + attention heatmap).  
  - Docstring updated: "ViT model (phase4_replay_best)", "attention heatmap" (no ResNet/Grad-CAM mention).

- **settings.py**  
  - `MODEL_NAME = 'phase4_replay_best.pth'`.  
  - No `IMAGE_MODEL_PATH`; single path from `ml_models/weights/`.

---

## 3. Database

| Item | Value |
|------|--------|
| Engine | PostgreSQL (`django.db.backends.postgresql`) |
| Name | `dmi_db` (config: `DB_NAME`) |
| User | `user` (config: `DB_USER`) |
| Password | `password` (config: `DB_PASSWORD`) |
| Host | `localhost` (local run) or `db` (docker-compose) |
| Port | `5432` |

- **analysis.models.AnalysisResult**  
  - Stores: `verdict`, `confidence`, `heatmap_path`, `explanation`, `metadata`, etc.  
  - Used for image (and video/text) results.

- **docker-compose.yml**  
  - `db`: postgres:15, `dmi_db`, user/password, port 5432.  
  - Start DB: `docker-compose up -d db`.

---

## 4. Frontend

| Item | Status |
|------|--------|
| **API call** | `lib/api.ts`: `analyzeImage()` → POST `/analysis/image/` with FormData `image`. |
| **Result display** | `app/results/[id]/page.tsx`: uses `verdict`, `confidence`, `heatmap_path` from API. |
| **Confidence** | Handles both 0–1 and 0–100 (e.g. `rawConfidence > 1 ? round : round*100`). Backend sends 0–1 → frontend shows % correctly. |
| **Heatmap** | Image URL: `http://127.0.0.1:8000/media/${heatmapPath}`. Backend returns relative path (e.g. `heatmaps/xxx_heatmap.jpg`). |

No frontend code change needed for ViT; same response shape.

---

## 5. Files that still mention ResNet (no impact on image flow)

- **ml_models/image_detection/model.py** – Defines `DeepfakeImageDetector` (ResNet). **Not imported by inference.py**; image pipeline does not use it.
- **ml_models/image_detection/gradcam.py** – Grad-CAM for ResNet. **Not imported by inference.py**; image pipeline does not use it.
- **ml_models/image_detection/evaluate_vit.py**, **evaluate_phase3_kaggle.py** – Scripts for evaluation; not used by Django or frontend.

---

## 6. Checklist before test

1. **Model file**  
   `dmi-backend/ml_models/weights/phase4_replay_best.pth` exists.

2. **Database**  
   - Docker: `cd dmi-backend` → `docker-compose up -d db`  
   - Migrations: `python manage.py migrate`

3. **Backend**  
   - `cd dmi-backend` → `python manage.py runserver`  
   - First image request will load ViT (and may download HuggingFace config if needed).

4. **Frontend**  
   - Run dev server (e.g. `npm run dev` in dmi-frontend).  
   - Use Image Analyze → upload image → expect verdict, confidence %, and heatmap.

---

## 7. Conclusion

- **Image detection:** Only ViT (phase4_replay_best) is used; ResNet is removed from the image pipeline.
- **APIs:** `/api/analysis/image/` is correct and uses ViT + attention heatmap.
- **Database:** PostgreSQL and `AnalysisResult` are correctly configured and used.
- **Frontend:** Uses same API response; no change required for ViT.

You can run backend + frontend and test image analysis with the above checklist.
