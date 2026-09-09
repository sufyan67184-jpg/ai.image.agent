from ultralytics import YOLO
import numpy as np

class YOLOFeatureAgent:
    def __init__(self):
        # Upgrade to the small (s) or medium (m) standard model if local, 
        # or stick to 'yolov8n.pt' with optimized detection thresholds for Render free tier.
        self.model = YOLO("yolov8s.pt")

    def extract_features(self, image):
        # Lower the confidence threshold to 0.10 and allow multi-scale predictions
        results = self.model.predict(image, conf=0.10, imgsz=640)
        r = results[0]
        
        features = []
        seen_names = set()
        
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                
                # Avoid duplicate entries of the same object type in the list
                if name not in seen_names:
                    seen_names.add(name)
                    features.append({
                        "feature_name": name,
                        "confidence_score": f"{conf:.2f}"
                    })
                
        width, height = image.size
        img_np = np.array(image)
        avg_rgb = img_np.mean(axis=(0, 1)).astype(int).tolist()
        
        summary = f"High-efficiency scan detected {len(features)} distinct structural elements."
        
        return {
            "summary": summary,
            "image_metadata": {
                "width_pixels": width,
                "height_pixels": height,
                "format": image.format or "JPEG",
                "color_mode": image.mode
            },
            "color_profile": {
                "average_rgb": avg_rgb
            },
            "detailed_features": features
        }
