"""GPU-pod SAM perception microservice v2 — perception only, raw
observations out; all identity decisions live on the backend brain.
Self-contained: serving needs sam3 + torch + cv2 + numpy + fastapi +
insightface (worker), never `src/`."""
