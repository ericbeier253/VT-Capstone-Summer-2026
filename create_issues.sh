echo "Creating US-03c"
issue_url=$(gh issue create --title "US-03c: Set up hashing for all objects identified" --body "As a user, I want to set up hashing for all objects identified in a picture and record timestamp and location of the objects in database.

T1: Module creation
Each of the module below is part of the ‘vision’ library created from dissociating the batch process and rewriting it for a streaming pipeline (for one image). For maintainability, testability, performance and extensibility, the modular architecture described below was used.

T2: Unit Tests for modules
Run a test for each of the above modules. Make sure the environment is consistent as the batch process.

T3: Test on Aria headset
The streaming pipeline can only be run with Aria headset running the server-side app. Any issues at this stage have to be debugged but the modular stateless architecture helps to isolate issues quickly.

T4: Setup a DINOv2 endpoint for lightweight client machines
Develop a Python-based REST API microservice that initializes and encapsulates the DINOv2 transformer model. Host the microservice locally on a home PC equipped with a dedicated GPU. Configure a Cloudflare Tunnel to securely expose the local API to the public internet. Refactor the client-side streaming code to send cropped object images via multipart POST requests to the tunneled remote /embed endpoint.")
gh issue close "$issue_url"

echo "Creating US-002"
issue_url=$(gh issue create --title "US-002: Hear a sound blip from the Aria headset at the time of snapshot" --assignee "@me" --body "As a user, I want to hear a sound blip from the Aria headset at the time of snapshot so that I can receive feedback that it’s working.

T1: Investigate audio capabilities on the Aria device
Review the Project Aria Client SDK documentation to identify available audio playback or Text-to-Speech (TTS) capabilities native to the headset.

T2: Implement auditory feedback in the streaming client
Integrate the render_tts(\"beep\") command within the eyegaze_callback handler to provide immediate auditory feedback whenever a valid gaze dwell triggers a snapshot. Implement try-except error handling around the audio command so that a failure in the headset speaker does not crash the overall streaming pipeline.")
gh issue close "$issue_url"

echo "Creating US-03A"
issue_url=$(gh issue create --title "US-03A: Search and filter object logs" --body "As a user, I want to search and filter object logs.

T1: Evaluate database search strategies for object retrieval in chatbot_viewer.py. Investigated both SQL-style keyword filtering and semantic search approaches for retrieving object logs from Firestore. Implemented and tested each method against user queries, comparing retrieval accuracy and relevance. Based on the results, recalibrated the implementation to prioritize semantic search because it provided more accurate and context-aware object matches than traditional SQL-style filtering.

T2: Integrate Gemini LLM with semantic search results to process the semantically retrieved database results by combining the object's metadata with its associated scene description. Generated clear, conversational responses that explain where the object is located, leveraging the improved search relevance from the semantic retrieval pipeline to provide more accurate answers to user queries.")
gh issue close "$issue_url"

echo "Creating US-03B"
issue_url=$(gh issue create --title "US-03B: Wirelessly stream from the headset" --assignee "@me" --body "As a user I want to wirelessly stream from the headset.

T1: Network configuration for the Aria headset
Configure the Aria headset to connect to the local Wi-Fi network using the mobile companion app or SDK CLI tools.

T2: Client application connection update
Modify the client application's connection initialization to target the headset’s IP address over Wi-Fi instead of defaulting to a hardwired USB connection. Test and optimize the streaming bandwidth and latency to ensure the wireless connection can support the throughput required for real-time gaze and RGB data processing.

T3: Troubleshoot Aria companion mobile app IP reporting.
Troubleshoot the Aria companion mobile app self-reporting the wrong host IP. The implemented workaround was to retrieve the IP directly from the network router, though this requires a network admin role, limiting its scope to private Wi-Fi networks.

T4: Define custom streaming profile.
Define a custom streaming profile with a reduced framerate. This was necessary to mitigate thermal issues and avoid overheating the device during prolonged wireless streaming sessions.")
gh issue close "$issue_url"

echo "Creating US-03D"
issue_url=$(gh issue create --title "US-03D: Process and view bounding boxes in the streamlit viewer" --assignee "@me" --body "As a user I want to process and view bounding boxes in the streamlit viewer.

T1: Stream coordinates from the enrichment worker.
Update the enrichment worker to pass the generated bounding box coordinates back to the client application state alongside the image data.

T2: Overlay bounding boxes in the UI
Modify the Streamlit viewer to intercept these coordinates and use an image drawing library (Pillow) to dynamically draw bounding box overlays onto the displayed frames. Ensure the bounding box coordinates are correctly scaled and aligned relative to the Viewer's UI image resolution, allowing for accurate visual verification.")
gh issue close "$issue_url"

echo "Creating US-03E"
issue_url=$(gh issue create --title "US-03E: Stream and orchestrate directly to my phone" --assignee "@me" --body "As a user I want to stream and orchestrate directly to my phone (vice laptop) in tandem with cloud resources.

T1: Investigate mobile SDK and direct streaming feasibility.
Investigate the feasibility and SDK requirements for running a Python/C++ streaming client directly on an iOS or Android device.

T2: Analyze constraints and document blockers
Identify that current compute constraints and mobile SDK limitations prevent native phone orchestration. Determine that this requires either setting up a secondary relay node or a complete mobile-native app rewrite before it can be achieved (Currently Blocked).")
gh issue close "$issue_url"

echo "Creating US-03F"
issue_url=$(gh issue create --title "US-03F: Live narrate the object of intention" --assignee "@me" --body "As a user I want to live narrate the object of intention for low-vis use case.

T1: Integrate LLM descriptions with Text-to-Speech
Explore routing the LLM-generated object descriptions (from the enrichment step) back to the headset using the render_tts() SDK function to read the text aloud.

T2: Assess real-time narration viability and blockers
Determine that the cumulative latency of the cloud enrichment pipeline (image upload + Gemini inference + text return) creates too much delay for a seamless \"live\" narration experience. Conclude that optimization of the inference loop is required first (Currently Blocked, perhaps feasible with on-premise compute).")
gh issue close "$issue_url"

echo "Creating US-03G"
issue_url=$(gh issue create --title "US-03G: Create a data enhancement pipeline to enrich the firestore" --body "As a user I want to create a data enhancement pipeline to enrich the firestore with data on the scene.

T1: Implement automated image ingestion in backfill_enrichment.py and develop functionality to monitor the Google Cloud Storage bucket for new image blobs, retrieve each unprocessed image, and prepare it for analysis. We investigated running the pipeline in Google Cloud Functions but ultimately decided to run it locally with python.threads.

T2: Integrate Gemini-based scene enrichment in backfill_enrichment.py and integrate the Google Gemini API into backfill_enrichment.py using a custom prompt to analyze each image and extract the scene description, detected objects, object names, and object descriptions. Formatted the analysis as structured JSON and updated the corresponding Firestore documents, enriching the database with searchable scene metadata.")
gh issue close "$issue_url"

echo "Creating US-03H"
issue_url=$(gh issue create --title "US-03H: Graphical interface to search for objects in the database" --body "As a user I want a graphical interface to search for objects in the database.

T1: Develop the Streamlit graphical interface in chatbot_viewer.py by using the Streamlit library to allow users to interact with the object search system. Implemented a chat-based interface that accepts natural language queries, displays conversation history, and updates responses in real time to provide an intuitive experience.

T2: Integrate Gemini-powered response into the Streamlit interface. Connected the Streamlit interface to the Gemini-powered chatbot so users can ask for the location of objects using natural language. Process user queries, retrieve relevant object and scene information from Firestore through the chatbot backend, and display Gemini-generated responses in real time.")
gh issue close "$issue_url"

echo "Done!"
