from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from agent import YOLOFeatureAgent

app = FastAPI(title="AI Vision Agent", version="2.0")
agent = YOLOFeatureAgent()

@app.get("/pride-ai", response_class=HTMLResponse)
async def serve_ui():
    # Serves our clean ChatGPT-style user interface page automatically
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/extract-yolo")
async def extract_yolo_features(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        feature_data = agent.extract_features(image)
        return {"filename": file.filename, "agent_output": feature_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
