# Traffic-Violation-Detector
Traffic Violation Detector is a modular traffic violation pipeline that eliminates false positives using geometric gating. It runs a COCO pass to find vehicles, checks if helmet/triple detections overlap "rider zones," and reuses coordinates to crop and OCR license plates. Includes a Streamlit UI &amp; Docker.
