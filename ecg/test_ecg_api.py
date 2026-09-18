import os
import json
from fastapi.testclient import TestClient
from main_api import app

def test_predict_ecg_endpoint():
    client = TestClient(app)
    sample_path = "MLII/1 NSR/100m (0).mat"

    print("=" * 70)
    print("        FASTAPI ENDPOINT TEST: POST /predict/ecg        ")
    print("=" * 70)

    if not os.path.exists(sample_path):
        print(f"Error: Sample file '{sample_path}' not found.")
        return

    print(f"\n[1] Testing with file: '{sample_path}'")

    with open(sample_path, "rb") as f:
        response = client.post(
            "/predict/ecg",
            files={"file": ("100m (0).mat", f, "application/octet-stream")}
        )

    print(f"[2] HTTP Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    json_data = response.json()
    print("\n--- [3] RETURNED JSON RESPONSE FROM API ---")
    print(json.dumps(json_data, indent=2))

    # Assertions
    assert json_data["status"] == "success"
    assert json_data["predicted_class"] == "1 NSR"
    assert json_data["class_id"] == 0
    assert len(json_data["top_5_probabilities"]) == 5
    assert "explainability" in json_data
    assert "hrv_metrics" in json_data["explainability"]

    print("\n" + "=" * 70)
    print(" SUCCESS: FastAPI /predict/ecg endpoint passed all assertions!")
    print("=" * 70)

if __name__ == "__main__":
    test_predict_ecg_endpoint()
