from src.core.product_detector import YOLODetector
import cv2
from src.core.llm_info_extractor import LLMInfoExtractor
def crop_detection(image, bbox):
    """
    Crop the detection from the image.
    """
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def main():
    """
    Main function to run the product detection system.
    """
    image = cv2.imread('data/test_images/IMG_2230.jpeg')
    detector = YOLODetector()
    detections = detector.detect(image)
    extractor = LLMInfoExtractor()
    for detection in detections[:2]:
        cropped_image = crop_detection(image, detection['bbox'])
        product_info = extractor.extract(cropped_image)
        print(product_info)
        
if __name__ == "__main__":
    main()