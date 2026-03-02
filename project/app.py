import os
from pathlib import Path

import torch
from flask import Flask, render_template, request, send_from_directory
from PIL import Image

from model_def import GenderAgeVGG16, build_transforms


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH_CANDIDATES = [BASE_DIR / "model.pth", BASE_DIR.parent / "model.pth"]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenderAgeVGG16().to(device)
inference_image_size = 224

model_path = next((candidate for candidate in MODEL_PATH_CANDIDATES if candidate.exists()), None)

if model_path is not None:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        inference_image_size = int(checkpoint.get("image_size", 224))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model_loaded = True
else:
    model_loaded = False

transform = build_transforms(inference_image_size)


def build_personalized_insights(gender_label: str, age_value: int, confidence_percent: str):
    if age_value <= 19:
        age_group = "Teen / Early Youth"
        diagnostic = "Predicted young skin profile with generally higher sebum activity and occasional acne sensitivity."
        morning_routine = [
            "Use a gentle gel cleanser",
            "Apply lightweight oil-free moisturizer",
            "Apply broad-spectrum SPF 50 sunscreen",
        ]
        evening_routine = [
            "Cleanse after outdoor exposure",
            "Use non-comedogenic moisturizer",
            "Use salicylic acid spot-care only if needed",
        ]
    elif age_value <= 29:
        age_group = "Young Adult"
        diagnostic = "Predicted early-adult profile where hydration balance, UV protection, and prevention are key."
        morning_routine = [
            "Use mild cleanser",
            "Apply antioxidant serum (Vitamin C)",
            "Use moisturizer with hyaluronic acid",
            "Apply broad-spectrum SPF 50 sunscreen",
        ]
        evening_routine = [
            "Double cleanse if makeup/sunscreen was used",
            "Apply niacinamide or hydration serum",
            "Use barrier-repair moisturizer",
        ]
    elif age_value <= 39:
        age_group = "Adult (30s)"
        diagnostic = "Predicted adult profile with potential first fine lines, requiring hydration + collagen-support routine."
        morning_routine = [
            "Gentle cleanser",
            "Vitamin C serum",
            "Ceramide moisturizer",
            "Broad-spectrum SPF 50 sunscreen",
        ]
        evening_routine = [
            "Cleanser",
            "Retinoid (2 to 3 nights/week initially)",
            "Hydrating moisturizer",
        ]
    elif age_value <= 49:
        age_group = "Mature Adult (40s)"
        diagnostic = "Predicted mature profile with higher need for elasticity support, barrier care, and consistent sun protection."
        morning_routine = [
            "Cream cleanser",
            "Peptide or antioxidant serum",
            "Rich moisturizer",
            "Broad-spectrum SPF 50 sunscreen",
        ]
        evening_routine = [
            "Cleanser",
            "Retinoid / peptide treatment",
            "Nourishing night cream",
        ]
    else:
        age_group = "Senior Skin Profile"
        diagnostic = "Predicted senior profile where deep hydration, barrier restoration, and gentle active use are recommended."
        morning_routine = [
            "Hydrating non-foaming cleanser",
            "Moisture-lock serum (hyaluronic acid)",
            "Barrier cream with ceramides",
            "Broad-spectrum SPF 50 sunscreen",
        ]
        evening_routine = [
            "Gentle cleanser",
            "Low-strength retinoid or peptide cream",
            "Occlusive moisturizer for overnight hydration",
        ]

    prognostic = (
        "With consistent sunscreen, hydration, sleep quality, and balanced diet over 8 to 12 weeks, "
        "skin texture and tone are expected to improve while visible aging progression can be slowed."
    )

    return {
        "age_group": age_group,
        "diagnostic": diagnostic,
        "prognostic": prognostic,
        "morning_routine": morning_routine,
        "evening_routine": evening_routine,
        "profile_note": f"Model confidence (male class): {confidence_percent}",
        "disclaimer": "This is an AI-based cosmetic wellness recommendation, not a medical diagnosis. Consult a dermatologist for clinical concerns.",
        "gender_context": f"Personalization context: {gender_label}, predicted age {age_value}",
    }


def predict(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        gender_prob, age_pred = model(input_tensor)

    gender_prob_value = float(gender_prob.squeeze().cpu().item())
    age_value = float(age_pred.squeeze().cpu().item())

    gender_label = "Male" if gender_prob_value >= 0.5 else "Female"
    age_value = max(0, min(80, round(age_value)))

    return gender_label, age_value, gender_prob_value


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    image_url = None

    if request.method == "POST":
        if not model_loaded:
            error = "model.pth not found. Train and place model.pth in project folder."
            return render_template("index.html", prediction=prediction, error=error, image_url=image_url)

        if "image" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", prediction=prediction, error=error, image_url=image_url)

        file = request.files["image"]
        if file.filename == "":
            error = "Please select an image file."
            return render_template("index.html", prediction=prediction, error=error, image_url=image_url)

        filename = os.path.basename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)
        image_url = f"/uploads/{filename}"

        gender, age, prob = predict(save_path)
        confidence_text = f"{prob * 100:.2f}%"
        prediction = {
            "gender": gender,
            "age": age,
            "confidence": confidence_text,
            "image_name": filename,
            "insights": build_personalized_insights(gender, age, confidence_text),
        }

    return render_template("index.html", prediction=prediction, error=error, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)
