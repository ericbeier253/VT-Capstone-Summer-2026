# VT-Capstone-Summer-2026: Group 1, Project Aria

This repository contains the codebase for our Project Aria eye-gaze tracking and streaming pipeline. It allows a wearer of the Project Aria glasses to stream live video and trigger photo captures based on eye gaze depth, automatically uploading the metadata and images to Google Cloud.

## Architecture & Database Design

We are utilizing a serverless Google Cloud architecture to store our data:
*   **Google Cloud Storage (GCS)**: Stores the raw triggered image captures (`.jpg`), neatly organized into folders by `run_id` (e.g. `run_20260711_150000/gaze_trigger.jpg`).
*   **Firestore (Native NoSQL)**: Stores the event metadata. The `gaze_events` collection contains documents with the following schema:
    *   `timestamp`: Time of the trigger.
    *   `depth`: The gaze depth measurement that triggered the event.
    *   `img_path`: A `gs://` URI referencing the corresponding image in GCS.
    *   `run_id`: The ID of the run session this event belongs to.
    *   `llm_analysis`: A structured JSON payload identifying the environment and itemized objects in the room under the gaze target.
*   **Gemini AI**: Asynchronously processes captured images in a background local worker thread to provide semantic scene and object analysis, pushing the enriched payload seamlessly to Firestore without blocking the camera stream.

## Setup Instructions

To run either the data collector or the viewer, you must configure your environment with the proper credentials.

1.  **Environment Variables**: Create a `.env` file in the root of the project with your specific project settings:
    ```env
    GCP_PROJECT="<YOUR_GCP_PROJECT_ID>"
    GCS_BUCKET="<YOUR_GCS_BUCKET_NAME>"
    GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/repo/.secrets/<YOUR_KEY_FILE>.json"
    GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
    ```
2.  **Service Account Key**: Place your Google Cloud service account JSON key inside the `.secrets/` directory. *(Note: The `.secrets/` folder is ignored by git to prevent credential leaks).*
3.  **Dependencies**: Install the required packages (using a virtual environment is highly recommended, but optional):
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage for the Glasses Wearer (Data Collection)

If you have the Project Aria glasses and are actively running a session to collect data, use the live stream script.

1. Activate your virtual environment.
2. Run the shell script with the `--cloud` flag to route data directly to GCP:
    ```bash
    ./run_live_stream.sh --cloud
    ```
*(If you omit the `--cloud` flag, it will attempt to default back to a local PostgreSQL instance).*

---

## Usage for Partners (The Viewer App)

If you do not have the glasses but want to view the data collections and images, you can run the Streamlit viewer application.

1. Ensure your `.env` and `.secrets/` folder are properly configured (see Setup Instructions).
2. Simply run the viewer launcher script:
    ```bash
    ./run_viewer.sh
    ```
3. A browser tab will automatically open. Use the dropdown to select a specific `run_id` to view the chronological stream of events and snapshots!