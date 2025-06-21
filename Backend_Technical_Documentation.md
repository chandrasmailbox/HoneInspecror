# HoneInspector Backend: Technical Documentation

---

## 1. Overview

The backend of HoneInspector is responsible for receiving video uploads, running AI/ML-based defect analysis on the videos, and returning structured results to the frontend. It is implemented in Python using FastAPI, is containerized via Docker, and stores data in MongoDB.

---

## 2. Technology Stack

- **Programming Language:** Python 3.10
- **Web Framework:** FastAPI
- **Async Task Management:** asyncio, ThreadPoolExecutor
- **Database:** MongoDB (accessed via Motor, an async MongoDB driver)
- **AI/ML Libraries:** 
  - OpenCV (for image processing, crack detection)
  - NumPy (for array operations)
  - PyTorch + Hugging Face Transformers (for CLIP model)
  - Pillow (image manipulation)
- **Other Utilities:** 
  - python-dotenv (env management)
  - Pydantic (data validation)
  - starlette (CORS middleware)
- **Containerization:** Docker
- **DevOps:** Example CI/CD in docs, recommends GitHub Actions

---

## 3. System Architecture & Component Interaction

- **API Endpoint:** `/api/`
  - Handles video upload and triggers analysis.
  - Returns JSON-formatted results.

- **Typical Flow:**
  1. **Frontend** uploads a video via REST API.
  2. **Backend** extracts frames from the video.
  3. For each frame:
     - Detects cracks using OpenCV edge detection.
     - Detects water damage via color analysis.
     - Optionally classifies defects using a CLIP model.
     - Draws bounding boxes on detected defects.
  4. Stores/upload results and images (assumed, see DB section).
  5. **Frontend** fetches and displays analysis.

---

## 4. Core Backend Logic

### 4.1. Video Upload & Frame Extraction

- Accepts video files in the API.
- Extracts frames at regular intervals using OpenCV.

### 4.2. Defect Detection

#### a) Crack Detection

- Uses OpenCV edge detection (e.g., Canny) on each frame.
- Post-processes edges to identify crack-like patterns.

#### b) Water Damage Detection

- Analyzes frame colors and patterns.
- Flags frames with color anomalies typical of water damage.

#### c) CLIP Model Classification

- Optional: Uses OpenAI CLIP model (via Hugging Face Transformers) to classify frames based on defect prompts.
- Model and processor are loaded globally at startup.

#### d) Bounding Boxes

- For each detected defect, computes bounding box coordinates.
- Annotates frames with colored boxes (Pillow).

### 4.3. Asynchronous Processing

- Uses asyncio and ThreadPoolExecutor for concurrent video processing.
- Returns results only after all frames are analyzed.

### 4.4. Data Models (Pydantic)

- **DefectDetection:** Represents a detected defect (type, confidence, box, etc.).
- **DefectFrame:** Represents a frame with associated defects and metadata.

### 4.5. MongoDB Integration

- Connection initialized from `.env` variables.
- Stores users, videos, results (see assumed schema below).

#### Example ERD (Assumed)

- **User**: id, name, email
- **Video**: id, user_id, upload_time, file_path
- **AnalysisResult**: id, video_id, summary_json

---

## 5. API Documentation (Summary)

- **POST `/api/analyze`**: Upload a video, returns analysis results (frames, defects, summary).
- **GET `/api/`**: Health check or basic info.

**Authentication:** Not implemented yet; future enhancement will require tokens.

---

## 6. Deployment

- **Dockerized:** Multi-stage build for React frontend and Python backend.
- **Environment Variables:** Backend reads MongoDB credentials and DB names from `.env`.
- **Production:** Backend and frontend served via NGINX in the final container.

---

## 7. Security & Future Enhancements

- **Current:** No authentication; basic CORS handling.
- **Planned:** Auth via tokens, user management, admin features.

---

## 8. References

- See `PROJECT_ARTIFACTS.md` for SRS, user stories, use cases, and more.
- For full API details, see section 6 in the artifacts doc.

---

*This document summarizes the backend architecture and logic as implemented in the HoneInspector project as of June 2025, based on repository analysis and code inspection.*