"""
Smart Career Path — Flask Prediction API
POST /predict  → accepts 27 boolean answers, returns top 3 career tracks
GET  /health   → uptime check
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import re
import os

app = Flask(__name__)

# ── Model definition (must match training exactly) ────────────────────────
class CareerMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ── Load artifacts once at startup ────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

tfidf = joblib.load(os.path.join(ARTIFACTS, "tfidf_vectorizer.pkl"))
le    = joblib.load(os.path.join(ARTIFACTS, "label_encoder.pkl"))

with open(os.path.join(ARTIFACTS, "model_meta.json")) as f:
    meta = json.load(f)

device = torch.device("cpu")
model  = CareerMLP(meta["input_dim"], meta["num_classes"]).to(device)
model.load_state_dict(
    torch.load(os.path.join(ARTIFACTS, "best_model.pt"), map_location=device)
)
model.eval()

# ── 27 skills in EXACT same order as the questions in the DB ─────────────
SKILLS = [
    "Web Development",
    "Mobile App Development",
    "Artificial Intelligence (AI) and Machine Learning",
    "Problem Solving and Analysis",
    "Cybersecurity",
    "Operating Systems and Networking",
    "Database Development",
    "Data Analysis and Visualization",
    "API Testing",
    "Performance Testing",
    "Statistical Analysis",
    "Deep Learning",
    "Machine Learning",
    "Data Engineering",
    "Cloud Computing",
    "Blockchain",
    "System Design",
    "Project Management",
    "Game Development",
    "Network Security",
    "Graphic Design",
    "UI/UX Knowledge",
    "Internet of Things",
    "Big Data Technologies",
    "Image Processing",
    "Feature Engineering",
    "Software Quality Testing"
]

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9,\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    answers = data.get("answers", [])

    if len(answers) != 27:
        return jsonify({
            "error": f"Expected 27 boolean answers, got {len(answers)}."
        }), 400

    # Extract skill names where answer is True
    yes_skills = [SKILLS[i] for i, val in enumerate(answers) if val]

    if not yes_skills:
        # No skills selected — return empty (backend handles this gracefully)
        return jsonify({"top3": []}), 200

    skills_text = clean(", ".join(yes_skills))

    vec    = tfidf.transform([skills_text]).toarray().astype(np.float32)
    tensor = torch.tensor(vec).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

    ranked = sorted(
        [(le.classes_[i], float(probs[i])) for i in range(len(le.classes_))],
        key=lambda x: -x[1]
    )

    top3 = [career for career, _ in ranked[:3]]
    return jsonify({"top3": top3}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
