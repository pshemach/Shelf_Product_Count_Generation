from src.core.product_detector import ProductDetector
import cv2

def main():
    """
    Main function to run the product detection system.
    """
    image = cv2.imread('data/test_images/IMG_2230.jpeg')
    detector = ProductDetector()
    detections = detector.detect_products(image)
    print(detections)

if __name__ == "__main__":
    main()