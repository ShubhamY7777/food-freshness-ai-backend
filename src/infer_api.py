from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as T
from PIL import Image
import io

from src.model import MultiTaskModel

# -------------------- APP --------------------
app = FastAPI(title="AI Food Freshness Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- LOAD MODEL --------------------
model = MultiTaskModel(num_classes=23)
model.load_state_dict(
    torch.load("checkpoints/best_model.pt", map_location=device)
)
model.to(device)
model.eval()

# -------------------- TRANSFORMS --------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- LABEL MAP --------------------
LABEL_MAP = {
    0: "freshapples",
    1: "freshbanana",
    2: "freshbittergroud",
    3: "freshcapsicum",
    4: "freshcucumber",
    5: "freshokra",
    6: "freshoranges",
    7: "freshpotato",
    8: "freshtomato",
    9: "rottenapples",
    10: "rottenbanana",
    11: "rottenbittergroud",
    12: "rottencapsicum",
    13: "rottencucumber",
    14: "rottenokra",
    15: "rottenoranges",
    16: "rottenpotato",
    17: "rottentomato",
    18: "freshpatato",
    19: "freshtamto",
    20: "rottenpatato",
    21: "rottentamto",
    22: "nonfood"
}

# -------------------- TEMPERATURE MAP --------------------
TEMPERATURE_MAP = {
    "freshapples": {"temp": "0-4°C", "multiplier": 1.5},
    "freshbanana": {"temp": "12-14°C", "multiplier": 1.3},
    "freshpotato": {"temp": "7-10°C", "multiplier": 1.4},
    "freshtomato": {"temp": "8-12°C", "multiplier": 1.3},
    "freshcucumber": {"temp": "8-10°C", "multiplier": 1.4},
    "freshokra": {"temp": "8-10°C", "multiplier": 1.4},
    "freshoranges": {"temp": "4-8°C", "multiplier": 1.6},
}

# -------------------- BASE DAYS FUNCTION --------------------
def estimate_base_days(label: str):
    if "fresh" in label:
        return 5
    elif "rotten" in label:
        return 0
    elif "nonfood" in label:
        return 0
    else:
        return 2

# -------------------- INFERENCE API --------------------
@app.post("/infer")
async def infer(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    x = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        reg_out, cls_out = model(x)

    # Classification
    cls_idx = int(torch.argmax(cls_out, dim=1).item())
    confidence_score = float(torch.softmax(cls_out, dim=1).max().item())

    state = LABEL_MAP.get(cls_idx, "unknown")

    # ---------------- ORIGINAL DAYS ----------------
    original_days = estimate_base_days(state)

    # ---------------- TEMPERATURE OPTIMIZATION ----------------
    recommended_temperature = "Not Required"
    improved_days = original_days

    if "fresh" in state and state in TEMPERATURE_MAP:
        multiplier = TEMPERATURE_MAP[state]["multiplier"]
        recommended_temperature = TEMPERATURE_MAP[state]["temp"]
        improved_days = int(original_days * multiplier)

    elif "rotten" in state:
        recommended_temperature = "Discard Immediately"
        improved_days = 0

    elif state == "nonfood":
        recommended_temperature = "Not Applicable"
        improved_days = 0

    # Format confidence
    confidence = f"{round(confidence_score * 100, 2)}%"

    return {
        "state": state,
        "confidence": confidence,
        "original_days_left": original_days,
        "recommended_temperature": recommended_temperature,
        "improved_shelf_life_days": improved_days
    }

# -------------------- RUN SERVER --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.infer_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )