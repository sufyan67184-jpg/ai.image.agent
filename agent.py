from ultralytics import YOLO
from PIL import Image
import numpy as np

class YOLOFeatureAgent:
    def __init__(self):
        # Loads the YOLOv8 nano model
        self.model = YOLO("yolov8n.pt")

    def extract_features(self, image: Image.Image) -> dict:
        # 1. Image Technical Metadata
        width, height = image.size
        format_type = image.format or "JPEG"
        mode = image.mode

        # 2. Color Profile Analysis (Average RGB)
        img_thumb = image.resize((50, 50))
        np_img = np.array(img_thumb)
        avg_color = np_img.mean(axis=(0, 1)).tolist()

        # 3. Full Object & Feature Detection
        results = self.model(image)
        result = results[0]
        
        detected_objects = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.model.names[int(cls_id)]
                detected_objects.append({
                    "feature_name": class_name,
                    "confidence_score": f"{round(float(conf) * 100, 1)}%",
                    "bounding_box": {
                        "xmin": round(float(box[0]), 1),
                        "ymin": round(float(box[1]), 1),
                        "xmax": round(float(box[2]), 1),
                        "ymax": round(float(box[3]), 1)
                    }
                })

        return {
            "summary": f"Extracted full technical and visual features: found {len(detected_objects)} object(s).",
            "image_metadata": {
                "width_pixels": width,
                "height_pixels": height,
                "format": format_type,
                "color_mode": mode
            },
            "color_profile": {
                "average_rgb": [round(c, 1) for c in avg_color]
            },
            "detailed_features": detected_objects
        }