from ultralytics import YOLO
import numpy as np

class YOLOFeatureAgent:
    def __init__(self):
        # Load the lightweight open-vocabulary YOLO-World model
        self.model = YOLO("yolov8s-world.pt")
        
        # Dynamically set any custom targets, objects, or your name to recognize them instantly!
        self.model.set_classes(["person", "face", "laptop", "phone", "sufyan"])

    def extract_features(self, image):
        # Use a lower confidence threshold (conf=0.15) to catch finer or subtle objects
        results = self.model.predict(image, conf=0.15)
        r = results[0]
        
        features = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                features.append({
                    "feature_name": name,
                    "confidence_score": f"{conf:.2f}"
                })
                
        width, height = image.size
        img_np = np.array(image)
        avg_rgb = img_np.mean(axis=(0, 1)).astype(int).tolist()
        
        summary = f"Open-vocabulary scan identified {len(features)} target elements."
        
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
