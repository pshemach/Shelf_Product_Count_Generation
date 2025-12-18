"""
System Status Checker

Quick check of the exact product matching system setup.
"""

from pathlib import Path
import json


def check_system_status():
    """Check if all components are properly set up."""
    
    print("=" * 60)
    print("Exact Product Matching System - Status Check")
    print("=" * 60)
    
    status = {
        'ready': True,
        'warnings': [],
        'errors': []
    }
    
    # Check reference images
    print("\n📁 Reference Images:")
    ref_dir = Path('data/reference_images')
    if ref_dir.exists():
        products = [d for d in ref_dir.iterdir() if d.is_dir()]
        total_images = sum(len(list(p.glob('*.jpg')) + list(p.glob('*.jpeg')) + list(p.glob('*.png'))) 
                          for p in products)
        print(f"  ✓ Directory exists: {ref_dir}")
        print(f"  ✓ Products: {len(products)} SKUs")
        print(f"  ✓ Total images: {total_images}")
        
        if len(products) == 0:
            status['errors'].append("No product folders found in reference_images/")
            status['ready'] = False
        elif total_images < 10:
            status['warnings'].append(f"Only {total_images} reference images. Recommend 5-10 per product.")
    else:
        print(f"  ✗ Directory not found: {ref_dir}")
        status['errors'].append("Reference images directory missing")
        status['ready'] = False
    
    # Check test images
    print("\n📸 Test Images:")
    test_dir = Path('data/test_images')
    if test_dir.exists():
        test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.jpeg')) + list(test_dir.glob('*.png'))
        print(f"  ✓ Directory exists: {test_dir}")
        print(f"  ✓ Test images: {len(test_images)}")
        
        if len(test_images) == 0:
            status['warnings'].append("No test images found")
    else:
        print(f"  ✗ Directory not found: {test_dir}")
        status['warnings'].append("Test images directory missing")
    
    # Check YOLO model
    print("\n🎯 YOLO Detection Model:")
    yolo_path = Path('data/best.pt')
    if yolo_path.exists():
        size_mb = yolo_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model exists: {yolo_path}")
        print(f"  ✓ Size: {size_mb:.1f} MB")
    else:
        print(f"  ✗ Model not found: {yolo_path}")
        status['warnings'].append("YOLO model missing - will use full image for demo")
    
    # Check trained classifier
    print("\n🧠 Trained Product Classifier:")
    model_path = Path('models/best_model.pth')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model exists: {model_path}")
        print(f"  ✓ Size: {size_mb:.1f} MB")
        print(f"  ✓ READY for accurate matching!")
    else:
        print(f"  ✗ Model not found: {model_path}")
        print(f"  ⚠ Run 'python train_model.py' to train the model")
        status['warnings'].append("Classifier not trained - low accuracy expected")
        status['ready'] = False
    
    # Check embedding index
    print("\n🔍 Embedding Index:")
    index_path = Path('data/product_embeddings.faiss')
    metadata_path = Path('data/product_embeddings_metadata.pkl')
    
    if index_path.exists() and metadata_path.exists():
        size_kb = index_path.stat().st_size / 1024
        print(f"  ✓ Index exists: {index_path}")
        print(f"  ✓ Metadata exists: {metadata_path}")
        print(f"  ✓ Size: {size_kb:.1f} KB")
    else:
        print(f"  ✗ Index not found")
        print(f"  ℹ Will be created automatically on first run")
        status['warnings'].append("Embedding index not built - will build on first run")
    
    # Check product database
    print("\n📊 Product Database:")
    db_path = Path('data/product_database.json')
    if db_path.exists():
        with open(db_path, 'r') as f:
            db = json.load(f)
        print(f"  ✓ Database exists: {db_path}")
        print(f"  ✓ Products: {db.get('total_products', 'unknown')}")
        print(f"  ✓ Brands: {db.get('total_brands', 'unknown')}")
    else:
        print(f"  ✗ Database not found: {db_path}")
        print(f"  ℹ Will be created automatically on first run")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print(f"  ✗ PyTorch not installed")
        status['errors'].append("PyTorch missing - run: pip install torch torchvision")
        status['ready'] = False
    
    try:
        import cv2
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print(f"  ✗ OpenCV not installed")
        status['errors'].append("OpenCV missing - run: pip install opencv-python")
        status['ready'] = False
    
    try:
        import faiss
        print(f"  ✓ FAISS: installed")
    except ImportError:
        print(f"  ✗ FAISS not installed")
        status['errors'].append("FAISS missing - run: pip install faiss-cpu")
        status['ready'] = False
    
    try:
        from ultralytics import YOLO
        print(f"  ✓ Ultralytics: installed")
    except ImportError:
        print(f"  ✗ Ultralytics not installed")
        status['errors'].append("Ultralytics missing - run: pip install ultralytics")
        status['ready'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if status['errors']:
        print("\n❌ ERRORS:")
        for error in status['errors']:
            print(f"  • {error}")
    
    if status['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in status['warnings']:
            print(f"  • {warning}")
    
    if status['ready'] and not status['errors']:
        print("\n✅ System is READY!")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Check results and adjust thresholds if needed")
    else:
        print("\n❌ System NOT ready")
        print("\nNext steps:")
        if "Classifier not trained" in str(status['warnings']):
            print("  1. Run: python train_model.py  (IMPORTANT!)")
            print("  2. Wait for training to complete")
            print("  3. Run: python main.py")
        else:
            print("  1. Fix errors listed above")
            print("  2. Run: python train_model.py")
            print("  3. Run: python main.py")
    
    print("\n" + "=" * 60)
    
    return status


if __name__ == "__main__":
    check_system_status()
