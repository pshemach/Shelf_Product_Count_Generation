from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Union
from pathlib import Path
from src.exception import MerchandiserException
import sys  
from src.logs import logging

class YOLODetector:
    """
    A class to detect product in shelf images using YOLO model.
    """
    def __init__(self, yolo_model_path: str = 'data/models/best.pt'):
        self.model_path = Path(yolo_model_path)
        if not self.model_path.exists():
            logging.error(f"YOLO model not found at {self.model_path}")
            raise MerchandiserException(f"YOLO model not found at {self.model_path}", sys) from FileNotFoundError(f"YOLO model not found at {self.model_path}")
        self.yolo_model = YOLO(self.model_path)
        logging.info(f"YOLO model loaded successfully from {self.model_path}")
        
    def detect(self, 
               image: np.ndarray,
               confidence_threshold: float = 0.5
               ) -> List[Dict]:
        try:
            results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
  
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': round(confidence, 2),
                        'class_id': class_id,
                        'class_name': class_name
                    })
            logging.info(f"Processed {len(detections)} detections")
            return detections
        
        except Exception as e:
            logging.error(f"Error detecting products: {e}")
            raise MerchandiserException(e, sys) from e