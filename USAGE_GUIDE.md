# Exact Product Matching System - Usage Guide

## 📁 Project Structure

```
Shelf_Product_Count_Generation/
├── data/
│   ├── reference_images/      # Reference product images (organized by SKU)
│   │   ├── 1000/
│   │   ├── 1001/
│   │   └── ...
│   ├── test_images/           # Test shelf images
│   ├── best.pt               # YOLO detection model
│   ├── product_database.json  # Product metadata
│   ├── product_embeddings.faiss  # Embedding index
│   └── product_embeddings_metadata.pkl
├── models/                    # Trained models will be saved here
│   ├── best_model.pth
│   └── config.json
├── exact_product_matcher.py  # Core matching system
├── train_model.py            # Training script
└── main.py                   # Main application
```

## 🚀 Quick Start

### 1️⃣ First Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# OR with uv
uv pip install -r requirements.txt
```

### 2️⃣ Train the Model (Required for accuracy)

```bash
python train_model.py
```

**What happens:**

- Loads all reference images from `data/reference_images/`
- Trains the ExactProductMatcher neural network
- Saves best model to `models/best_model.pth`
- Takes ~10-30 minutes depending on your GPU/CPU

**Training Output:**

```
Configuration:
  reference_dir: data/reference_images
  batch_size: 16
  num_epochs: 100
  ...

Loading dataset...
Dataset: 42 images, 21 products, 1 brands

Epoch 1/100
----------------------------------------------------------------------
Training: 100%|████████| 3/3 [00:05<00:00, loss=2.1234, prod_acc=15.00%]
Train Product Accuracy: 15.00%
Val Product Accuracy: 20.00%
✓ Saved best model

...

Epoch 100/100
Train Product Accuracy: 95.00%
Val Product Accuracy: 92.00%
✓ Saved best model
```

### 3️⃣ Run Product Counting

```bash
python main.py
```

**What happens:**

- Loads trained model (if available)
- Builds/loads embedding index
- Detects products using YOLO
- Matches and counts exact products
- Shows detailed results

**Expected Output (with trained model):**

```
[Step 5] Matching first detected product...

  Match Result:
  ├─ SKU: 1003
  ├─ Brand: 10
  ├─ Classification Confidence: 0.956
  ├─ Embedding Similarity: 0.912
  ├─ Embedding Verified: True
  ├─ Final Confidence: 0.912
  └─ Matched: ✓

[Step 6] Counting all detected products...

  Product Counts:
    ├─ 1000 (Product_1000): 3 units (avg conf: 0.891)
    ├─ 1003 (Product_1003): 5 units (avg conf: 0.923)
    ├─ 1007 (Product_1007): 2 units (avg conf: 0.887)
    └─ 1015 (Product_1015): 4 units (avg conf: 0.901)
```

## 📊 Understanding the Results

### Confidence Scores

- **Classification Confidence**: How confident the neural network is

  - `>0.9`: Very confident
  - `0.7-0.9`: Good confidence
  - `0.5-0.7`: Moderate confidence
  - `<0.5`: Low confidence (untrained model or unclear image)

- **Embedding Similarity**: How similar to reference images

  - `>0.85`: Very similar (verified match)
  - `0.75-0.85`: Similar
  - `<0.75`: Not verified

- **Final Confidence**: Combined score
  - Used for counting decisions
  - Threshold: 0.6 (adjustable in code)

### Match Status

- ✓ **Matched**: Product identified and counted
- ✗ **Unmatched**: Below confidence threshold

## 🎯 Use Cases

### Use Case 1: Count Products in Shelf Image

```python
from exact_product_matcher import ExactProductCounter
import cv2

# Initialize
counter = ExactProductCounter(
    model_path='models/best_model.pth',
    index_path='data/product_embeddings.faiss',
    metadata_path='data/product_embeddings_metadata.pkl'
)

# Load shelf image
shelf_image = cv2.imread('data/test_images/IMG_2329.jpeg')

# Detect products (using YOLO)
from ultralytics import YOLO
yolo = YOLO('data/best.pt')
results = yolo(shelf_image)

# Extract crops
detected_products = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = shelf_image[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        detected_products.append(crop_rgb)

# Count products
counts = counter.count_products(detected_products)

print(f"Total products detected: {counts['total_detected']}")
print(f"Total products matched: {counts['total_matched']}")
print(f"Product counts: {counts['counts']}")
```

### Use Case 2: Match Single Product Image

```python
# Load single product image
product_img = cv2.imread('cropped_product.jpg')
product_rgb = cv2.cvtColor(product_img, cv2.COLOR_BGR2RGB)

# Match
result = counter.match_product(product_rgb)

if result['matched']:
    print(f"Product identified: {result['sku']}")
    print(f"Confidence: {result['final_confidence']:.2%}")
else:
    print("Product not matched")
```

### Use Case 3: Add New Products

1. **Create new folder** in `data/reference_images/`:

   ```
   data/reference_images/1021/  # New SKU
   ```

2. **Add reference images** (2-5 images recommended):

   ```
   data/reference_images/1021/
       image1.jpg
       image2.jpg
       image3.jpg
   ```

3. **Rebuild embedding index**:

   ```bash
   # Delete old index
   rm data/product_embeddings.faiss
   rm data/product_embeddings_metadata.pkl

   # Run main.py (will rebuild automatically)
   python main.py
   ```

4. **Retrain model** (for best accuracy):
   ```bash
   python train_model.py
   ```

## ⚙️ Configuration

### Adjust Confidence Thresholds

Edit `main.py`:

```python
result = counter.match_product(
    product_image,
    classification_threshold=0.6,  # Lower = more permissive
    embedding_threshold=0.75,      # Higher = stricter verification
    use_embedding_verification=True
)
```

### Training Parameters

Edit `train_model.py`:

```python
config = {
    'batch_size': 16,      # Larger = faster but needs more memory
    'num_epochs': 100,     # More = better accuracy (if not overfitting)
    'learning_rate': 0.0001,  # Lower = more stable training
    'train_split': 0.8     # 80% train, 20% validation
}
```

## 🔍 Troubleshooting

### Problem: Low Accuracy (< 50%)

**Solution:**

1. Train the model: `python train_model.py`
2. Add more reference images (5-10 per product)
3. Increase training epochs in `train_model.py`

### Problem: "No products matched"

**Possible causes:**

- Model not trained → Run `python train_model.py`
- Thresholds too high → Lower in `main.py`
- Poor quality images → Use clearer reference images

### Problem: Wrong Product Matches

**Solutions:**

- Enable embedding verification (already enabled by default)
- Add more diverse reference images
- Increase `embedding_threshold` to 0.85 or 0.90

### Problem: YOLO not detecting products

**Solutions:**

- Check if `data/best.pt` exists
- Retrain YOLO on your shelf images
- Adjust YOLO confidence threshold

## 📈 Performance Tips

### For Better Accuracy:

1. **More reference images**: 5-10 per product
2. **Diverse angles**: Front, side, tilted views
3. **Different lighting**: Bright, dim, natural light
4. **Train longer**: 100-200 epochs
5. **Use GPU**: 10-20x faster training

### For Faster Inference:

1. **Use GPU**: Enable CUDA
2. **Batch processing**: Process multiple products at once
3. **Smaller model**: Use EfficientNet-S instead of EfficientNet-M

## 📝 Next Steps

1. ✅ Train the model
2. ✅ Test on your shelf images
3. ✅ Adjust thresholds if needed
4. ✅ Add more products to reference database
5. ✅ Integrate into your workflow

## 🆘 Need Help?

- Check console output for detailed error messages
- Review confidence scores to diagnose issues
- Ensure reference images are clear and well-lit
- Verify YOLO model is detecting products correctly

---

**System Status:**

- ✓ Reference images: `data/reference_images/` (21 products)
- ✓ Test images: `data/test_images/IMG_2329.jpeg`
- ✓ YOLO model: `data/best.pt`
- ⏳ Trained model: Run `python train_model.py` first
