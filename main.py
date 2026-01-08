from src.core.product_detector import YOLODetector
import cv2
from src.core.llm_info_extractor import LLMInfoExtractor
from src.core.llm_matcher import LLMProductMatcher

def crop_detection(image, bbox):
    """
    Crop the detection from the image.
    """
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def main():
    """
    Main function to run the product detection system.
    """
    image = cv2.imread('data/output/shelf_stitched.jpg')
    detector = YOLODetector()
    detections = detector.detect(image)
    extractor = LLMInfoExtractor()
    product_infos = []
    for detection in detections[:2]:
        cropped_image = crop_detection(image, detection['bbox'])
        product_info = extractor.extract(cropped_image)
        product_infos.append(product_info)
    matcher = LLMProductMatcher()
    match_results = matcher.batch_match(product_infos)
    print(match_results)
        
if __name__ == "__main__":
    main()