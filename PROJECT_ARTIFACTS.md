# HoneInspector Project Documentation & Artifacts

---

## 1. Software Requirements Specification (SRS)

### 1.1 Functional Requirements

- **Video Upload & Analysis**
  - Users can upload videos (of house/property interiors/exteriors).
  - The system analyzes the uploaded video to detect house defects via AI.
  - Defect types include: Cracks, Water Damage, Mold, Paint Issues, Rust, Broken Tiles, Damaged Flooring, Other.
  - The app provides frame-by-frame analysis, highlighting defect locations with colored boxes.

- **Results Display**
  - After analysis, users get a summary: total defects, frames analyzed, high-confidence defects, severity rating.
  - The UI shows a legend for each defect type.
  - Users can click on frames to see exact defect locations and details.

- **Reanalysis**
  - Users can return to the upload screen to analyze another video.

### 1.2 Non-Functional Requirements

- **Performance:** Analysis should be completed in a reasonable time for standard-length videos.
- **Usability:** Clean, accessible UI; color-coded legends and tooltips.
- **Reliability:** File types validated before upload; error handling for unsupported files.
- **Scalability:** Should handle multiple concurrent users and video sizes typical for home inspections.
- **Security:** Accept only authenticated requests for analysis (future enhancement).
- **Portability:** Works on modern browsers; responsive design.

### 1.3 System Architecture Overview

- **Frontend:** React.js SPA (Single Page Application)
- **Backend:** Not covered in the provided code, but the frontend expects a REST API at `${BACKEND_URL}/api`
- **AI/ML Model:** Assumed to be in the backend, which processes videos and returns defect analysis.

### 1.4 User Roles and Permissions

- **Inspector/User:** Can upload videos, view analysis, and download results.
- **Admin:** (Future) Could manage users, view usage statistics, retrain models, etc.

### 1.5 External System Dependencies

- **Backend API:** Must be available and configured via `REACT_APP_BACKEND_URL`.
- **AI Model:** For defect detection (not included in frontend).
- **Node.js/NPM:** For running frontend.
- **Browser:** For accessing the webapp.

---

## 2. User Stories and Acceptance Criteria

### Feature: Video Upload and Analysis

#### As an inspector, I want to upload a property video so that I can detect and document defects automatically.

- **Acceptance Criteria**
  - The upload button accepts only video files.
  - Unsupported file types show an error.
  - After upload, analysis starts automatically.

### Feature: Results Visualization

#### As an inspector, I want to see a summary of detected defects so I can quickly assess property condition.

- **Acceptance Criteria**
  - The summary shows total defects, frames analyzed, and severity.
  - Defect types are color-coded and listed.

#### As an inspector, I want to click on individual frames to see where defects are located.

- **Acceptance Criteria**
  - Clicking a frame shows an image with colored boxes and defect details.
  - Each defect shows confidence score and description.

### Feature: Reanalysis

#### As an inspector, I want to return to the upload screen so I can analyze another video.

- **Acceptance Criteria**
  - "Back to Upload" button resets the app state.

---

## 3. Use Case Diagrams & Descriptions

### Use Case: Upload Video

- **Actor:** Inspector/User
- **Description:** User uploads a video file for defect analysis.
- **Precondition:** User is on the upload page.
- **Postcondition:** Video is sent to backend for analysis.

### Use Case: View Results

- **Actor:** Inspector/User
- **Description:** User sees a summary and breakdown of detected defects.
- **Precondition:** Video analysis is complete.
- **Postcondition:** User can review and download results.

### Use Case: Drill-down on Frame

- **Actor:** Inspector/User
- **Description:** User examines a specific frame for detailed defect locations and types.
- **Precondition:** Analysis results are displayed.
- **Postcondition:** User sees frame image with overlays.

```
[Inspector] ---> (Upload Video) ---> (System Analyzes) ---> (View Results) ---> (Drill-down on Frame)
```

---

## 4. System Architecture Document

### 4.1 High-Level Architecture

- **Frontend:** React.js
  - Components:
    - VideoUpload
    - ResultsDisplay
    - DefectLegend
    - FrameDetailModal

- **Backend:** REST API (assumed, not present in repo)
  - Receives video, runs AI analysis, returns JSON results.

### 4.2 Technology Stack

- **Frontend:** React, Tailwind CSS (for styling), Axios (for HTTP requests)
- **Backend:** (Assumed) Python (FastAPI/Flask), AI/ML model

### 4.3 Component Interaction

- VideoUpload → uploads video → Backend API
- Backend API → returns analysis JSON → ResultsDisplay
- ResultsDisplay → shows summary, frame list → FrameDetailModal (on click)

### 4.4 Database Schema

*(No schema in frontend. Assumed: backend stores videos, analysis results, users.)*

#### Example ER Diagram (Assumed)

- User (id, name, email, etc.)
- Video (id, user_id, upload_time, file_path)
- AnalysisResult (id, video_id, summary_json)

---

## 5. Installation Guide

### 5.1 Environment Setup

- Install [Node.js](https://nodejs.org/)
- Clone the repository:  
  `git clone https://github.com/chandrasmailbox/HoneInspecror.git`
- Go to the frontend directory:  
  `cd HoneInspecror/frontend`

### 5.2 Install Dependencies

```bash
npm install
```

### 5.3 Run Locally

```bash
npm start
```
- Opens at [http://localhost:3000](http://localhost:3000)
- Ensure backend API is running and available at the URL set in `.env` as `REACT_APP_BACKEND_URL`.

### 5.4 Production Build

```bash
npm run build
```
- Output in `frontend/build`

---

## 6. API Documentation

### Video Analysis Endpoint

- **URL:** `${BACKEND_URL}/api/analyze`
- **Method:** `POST`
- **Request:** `multipart/form-data` with video file
- **Response:** JSON containing defect analysis

#### Example Request (curl)

```bash
curl -X POST "${BACKEND_URL}/api/analyze" \
  -F "file=@/path/to/video.mp4"
```

#### Example Response

```json
{
  "summary": {
    "total_defects_found": 8,
    "frames_analyzed": 120,
    "high_confidence_detections": 5,
    "severity": "high",
    "defect_types": ["Cracks", "Mold", "Water Damage"]
  },
  "defects_found": [
    {
      "frame_number": 10,
      "frame_image": "<base64>",
      "defects": [
        {
          "type": "Cracks",
          "confidence": 0.97,
          "description": "Horizontal crack in ceiling",
          "boxes": [{ "x": 12, "y": 34, "width": 20, "height": 5 }]
        }
      ]
    }
  ]
}
```

#### Status Codes

- 200: Success
- 400: Invalid file
- 500: Analysis error

### Authentication

- Not implemented in sample; future enhancement to require auth tokens.

---

## 7. Admin/User Guide

### How to Use

1. Open the application in a browser.
2. Click the upload area and select a video file (e.g., an MP4).
3. Wait for analysis to complete; view results.
4. Click frames to see detailed defect locations.

### Troubleshooting

- **Upload fails:** Ensure backend API is running and `REACT_APP_BACKEND_URL` is set.
- **Unsupported file:** Only video files are accepted.
- **Slow analysis:** Video files should be of reasonable length.

### FAQ

- *What types of defects can be detected?*  
  Cracks, Water Damage, Mold, Paint Issues, Rust, Broken Tiles, Damaged Flooring, Other.

---

## 8. Test Strategy & Test Cases

### 8.1 Test Strategy

- **Unit Tests:** For React components (with Jest/React Testing Library).
- **Integration Tests:** Upload flow, result display.
- **End-to-End Tests:** (Suggested: Cypress) Simulate a user uploading and analyzing a video.

### 8.2 Sample Test Cases

| Test Case                        | Steps                                             | Expected Result                         |
|----------------------------------|--------------------------------------------------|-----------------------------------------|
| Upload valid video               | Upload .mp4 file                                 | Analysis completes, results shown       |
| Upload invalid file              | Upload .jpg file                                 | Error displayed                        |
| Click frame in results           | Click a frame in summary                         | Modal shows defects for frame           |
| Back to upload                   | Click "Back to Upload" button                    | Upload screen shown                     |

### 8.3 Tools

- **Jest** for unit testing
- **React Testing Library** for component tests
- (Recommended: **Cypress** for E2E)

---

## 9. CI/CD Pipeline Explanation

- **Build:** `npm run build` on PR/merge
- **Test:** `npm test` runs unit/integration tests
- **Deploy:** Uploads production build to host (e.g., Vercel, Netlify)

### Example: GitHub Actions (not in repo, but recommended)

```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm install
      - run: npm test
      - run: npm run build
```

---

## 10. Contribution Guidelines

- **Fork** the repository and clone your fork.
- **Create a branch** for your feature (`git checkout -b feature/my-feature`)
- **Commit** descriptive messages.
- **Push** your branch and open a **Pull Request**.
- **Code Formatting:** Use Prettier (suggested), ESLint for linting.
- **Reviews:** At least one approval required before merging.

---

## 11. Glossary and Acronyms

| Term         | Definition                                                        |
|--------------|-------------------------------------------------------------------|
| Defect       | Anomaly detected in the video (e.g., crack, mold, water damage)   |
| Frame        | Single image extracted from the uploaded video                    |
| Confidence   | Likelihood (0-1) that a detected defect is correct                |
| Severity     | Overall quality issue rating, based on defects found              |
| AI/ML        | Artificial Intelligence / Machine Learning                        |
| Inspector    | User uploading and analyzing property video                       |
| API          | Application Programming Interface                                 |
| SPA          | Single Page Application                                           |
| CRUD         | Create, Read, Update, Delete (standard backend operations)        |

---

## Other Relevant Artifacts

- **Defect Color Legend:**  
  - Cracks: Red  
  - Water Damage: Blue  
  - Mold: Green  
  - Paint Issues: Yellow  
  - Rust: Orange  
  - Broken Tiles: Purple  
  - Damaged Flooring: Pink  
  - Other: Gray

---

*This documentation is based on the code and structure found in the HoneInspector repository as of June 2025. Please update as features evolve.*