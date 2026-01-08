import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.product_detector import YOLODetector
import cv2
import os

os.makedirs('data/output/cropped_detections', exist_ok=True)
image_path = r'data\shelf_images\IMG_2227.jpeg'

def crop_detection(image, bbox):
    """
    Crop the detection from the image.
    """
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def main():
    """
    Main function to run the product detection system.
    """
    image = cv2.imread(image_path)
    detector = YOLODetector()
    detections = detector.detect(image)
    for id, detection in enumerate(detections):
        cropped_image = crop_detection(image, detection['bbox'])
        cv2.imwrite(f'data/output/cropped_detections/{id}.jpg', cropped_image)
        
if __name__ == "__main__":
    main()