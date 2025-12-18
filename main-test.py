"""
Shelf Product Count Generation - Main Application

This script demonstrates how to use ExactProductCounter for:
1. Building embedding index from reference images
2. Detecting products in shelf images using YOLO
3. Matching and counting exact products
"""

import cv2
import numpy as np
from pathlib import Path
from exact_product_matcher import ExactProductCounter
from ultralytics import YOLO


def main():
    print("=" * 60)
    print("Shelf Product Count Generation")
    print("=" * 60)
    
    # ========================================
    # Step 1: Initialize the ExactProductCounter
    # ========================================
    print("\n[Step 1] Initializing ExactProductCounter...")
    
    # Check if trained model exists
    trained_model_path = Path('models/best_model.pth')
    if trained_model_path.exists():
        print("Found trained model!")
        model_path = str(trained_model_path)
    else:
        print("No trained model found - using untrained model")
        print("Run 'python train_model.py' to train the model first")
        model_path = None
    
    counter = ExactProductCounter(
        model_path=model_path,
        index_path='data/product_embeddings.faiss',
        metadata_path='data/product_embeddings_metadata.pkl',
        db_path='data/product_database.json'
    )
    
    print(f"✓ Loaded {len(counter.product_db.products)} products from reference database")
    print(f"✓ Brands: {len(counter.product_db.get_brands())}")
    
    # ========================================
    # Step 2: Build Embedding Index (First time only)
    # ========================================
    index_path = Path('data/product_embeddings.faiss')
    
    if not index_path.exists():
        print("\n[Step 2] Building embedding index from reference images...")
        print("This will extract embeddings from all images in data/reference_images/")
        
        counter.build_embedding_index(
            save_index_path='data/product_embeddings.faiss',
            save_metadata_path='data/product_embeddings_metadata.pkl'
        )
        print("✓ Embedding index built and saved!")
    else:
        print("\n[Step 2] Embedding index already exists, skipping build...")
    
    # ========================================
    # Step 3: Load Test Image
    # ========================================
    print("\n[Step 3] Loading test image...")
    
    test_image_path = 'data/test_images/IMG_2329.jpeg'
    shelf_image = cv2.imread(test_image_path)
    
    if shelf_image is None:
        print(f"✗ Error: Could not load image from {test_image_path}")
        return
    
    print(f"✓ Loaded test image: {test_image_path}")
    print(f"  Image size: {shelf_image.shape[1]}x{shelf_image.shape[0]}")
    
    # ========================================
    # Step 4: Detect Products using YOLO
    # ========================================
    print("\n[Step 4] Detecting products in shelf image...")
    
    # Load YOLO model (assuming you have a trained model)
    yolo_model_path = 'data/best.pt'
    
    if not Path(yolo_model_path).exists():
        print(f"✗ YOLO model not found at {yolo_model_path}")
        print("  Using entire image as single product for demonstration...")
        
        # For demo: use entire image as one product
        detected_products = [cv2.cvtColor(shelf_image, cv2.COLOR_BGR2RGB)]
        
    else:
        # Use YOLO for detection
        yolo_model = YOLO(yolo_model_path)
        results = yolo_model(shelf_image)
        
        detected_products = []
        
        # Extract cropped products from detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop product
                product_crop = shelf_image[y1:y2, x1:x2]
                product_crop_rgb = cv2.cvtColor(product_crop, cv2.COLOR_BGR2RGB)
                detected_products.append(product_crop_rgb)
        
        print(f"✓ Detected {len(detected_products)} products")
        
        # Store results for later use (for annotation)
        yolo_results = results
    
    # ========================================
    # Step 5: Match Single Product (Example)
    # ========================================
    if detected_products:
        print("\n[Step 5] Matching first detected product...")
        
        first_product = detected_products[0]
        result = counter.match_product(
            first_product,
            classification_threshold=0.6,  # Lower threshold since model is untrained
            embedding_threshold=0.75,
            use_embedding_verification=True
        )
        
        print(f"\n  Match Result:")
        print(f"  ├─ SKU: {result['sku']}")
        print(f"  ├─ Brand: {result['brand']}")
        print(f"  ├─ Classification Confidence: {result['classification_confidence']:.3f}")
        print(f"  ├─ Embedding Similarity: {result['embedding_similarity']:.3f}")
        print(f"  ├─ Embedding Verified: {result['embedding_verified']}")
        print(f"  ├─ Final Confidence: {result['final_confidence']:.3f}")
        print(f"  └─ Matched: {'✓' if result['matched'] else '✗'}")
    
    # ========================================
    # Step 6: Count All Products
    # ========================================
    print("\n[Step 6] Counting all detected products...")
    
    counts_result = counter.count_products(
        detected_products,
        classification_threshold=0.6,  # Lower threshold since model is untrained
        embedding_threshold=0.75,
        use_embedding_verification=True
    )
    
    print(f"\n  Count Summary:")
    print(f"  ├─ Total Detected: {counts_result['total_detected']}")
    print(f"  ├─ Total Matched: {counts_result['total_matched']}")
    print(f"  └─ Unmatched: {counts_result['total_detected'] - counts_result['total_matched']}")
    
    print(f"\n  Product Counts:")
    if counts_result['counts']:
        for sku, count in sorted(counts_result['counts'].items()):
            avg_conf = counts_result['avg_confidence'][sku]
            product_name = counter.product_db.get_product(sku)['name']
            print(f"    ├─ {sku} ({product_name}): {count} units (avg conf: {avg_conf:.3f})")
    else:
        print("    └─ No products matched")
    
    # ========================================
    # Step 7: Detailed Results
    # ========================================
    print("\n[Step 7] Detailed match results:")
    print(f"  {'ID':<4} {'SKU':<8} {'Brand':<8} {'Cls Conf':<10} {'Emb Sim':<10} {'Final':<10} {'Matched':<8}")
    print("  " + "-" * 70)
    
    for match in counts_result['all_matches'][:10]:  # Show first 10
        print(f"  {match['detection_id']:<4} "
              f"{match['sku']:<8} "
              f"{match['brand']:<8} "
              f"{match['classification_confidence']:<10.3f} "
              f"{match['embedding_similarity']:<10.3f} "
              f"{match['final_confidence']:<10.3f} "
              f"{'✓' if match['matched'] else '✗':<8}")
    
    # ========================================
    # Step 8: Save Annotated Image
    # ========================================
    print("\n[Step 8] Saving annotated image with detections...")
    
    # Create output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Create annotated image
    annotated_image = shelf_image.copy()
    
    # Check if we have YOLO results
    if 'yolo_results' in locals() and yolo_results:
        for i, box in enumerate(yolo_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get match result for this detection
            match_result = counts_result['all_matches'][i]
            sku = match_result['sku']
            confidence = match_result['final_confidence']
            matched = match_result['matched']
            
            # Choose color based on match status
            if matched:
                color = (0, 255, 0)  # Green for matched
            else:
                color = (0, 0, 255)  # Red for unmatched
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Prepare text
            label = f"ID:{i} SKU:{sku}"
            conf_text = f"Conf:{confidence:.2f}"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)
            
            # Draw background rectangles for text
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_h - conf_h - 15), 
                         (x1 + max(label_w, conf_w) + 10, y1), 
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_image, label, 
                       (x1 + 5, y1 - conf_h - 10), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated_image, conf_text, 
                       (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Add summary text at top
        summary_text = f"Detected: {len(detected_products)} | Matched: {counts_result['total_matched']} | Unmatched: {len(detected_products) - counts_result['total_matched']}"
        cv2.putText(annotated_image, summary_text, 
                   (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Save annotated image
        output_path = output_dir / 'annotated_shelf_image.jpg'
        cv2.imwrite(str(output_path), annotated_image)
        print(f"  ✓ Saved annotated image: {output_path}")
        
        # Also save a smaller version for quick preview
        scale_percent = 30  # Scale to 30% for preview
        width = int(annotated_image.shape[1] * scale_percent / 100)
        height = int(annotated_image.shape[0] * scale_percent / 100)
        preview = cv2.resize(annotated_image, (width, height))
        preview_path = output_dir / 'annotated_shelf_image_preview.jpg'
        cv2.imwrite(str(preview_path), preview)
        print(f"  ✓ Saved preview image: {preview_path}")
    else:
        print("  ✗ No YOLO detections available to annotate")
    
    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    
    # ========================================
    # Important Notes
    # ========================================
    print("\n📝 NOTES:")
    if not trained_model_path.exists():
        print("  ⚠ Model is UNTRAINED - low accuracy expected")
        print("  ➜ To train the model:")
        print("     1. Run: python train_model.py")
        print("     2. Wait for training to complete (~30-100 epochs)")
        print("     3. Run main.py again with trained model")
    else:
        print("  ✓ Using trained model - better accuracy expected")
        print("  ✓ Embedding verification enabled")
        print("  ➜ To retrain: python train_model.py")
    

if __name__ == "__main__":
    main()