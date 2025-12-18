"""
Shelf Product Identification System with Reference Products
Uses object detection + OCR + reference matching
"""

import cv2
import numpy as np
import requests
import json
import base64
import os
from typing import List, Dict, Tuple, Literal
import unicodedata
import time
from ultralytics import YOLO
from rapidfuzz import fuzz, process
from llm_verifier_unified import UnifiedProductMatchVerifier

LLMProvider = Literal["openai", "anthropic"]

class ShelfProductIdentifier:
    def __init__(
        self,
        openai_api_key: str,
        yolo_model_path: str = 'yolov8n.pt',
        llm_provider: LLMProvider = "openai",
        llm_model: str = None,
        ocr_provider: LLMProvider = "openai",
        ocr_model: str = None
    ):
        """
        Initialize the shelf product identifier.
        
        Args:
            openai_api_key: OpenAI API key (required for OpenAI OCR/verification)
            yolo_model_path: Path to YOLO model weights (default: yolov8n.pt)
            llm_provider: LLM provider for match verification ("openai" or "anthropic")
            llm_model: Specific LLM model for verification (optional, uses provider defaults)
            ocr_provider: LLM provider for OCR text extraction ("openai" or "anthropic")
            ocr_model: Specific LLM model for OCR (optional, uses provider defaults)
        """
        self.openai_api_key = openai_api_key
        self.reference_products = {}  # Store reference products
        self.last_api_call_time = 0
        self.min_delay_between_calls = 0.5  # Minimum 0.5 second between API calls
        
        # Load YOLO model
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        print("YOLO model loaded successfully")
        
        # Initialize LangChain-based OCR with selected provider
        print(f"Initializing {ocr_provider.upper()} OCR...")
        self.ocr_verifier = UnifiedProductMatchVerifier(
            provider=ocr_provider,
            model_name=ocr_model or ("gpt-4o-mini" if ocr_provider == "openai" else "claude-3-5-haiku-20241022"),
            temperature=0
        )
        print(f"OCR initialized: {self.ocr_verifier.get_info()}")
        
        # Initialize LangChain-based LLM verifier with selected provider
        print(f"Initializing {llm_provider.upper()} LLM verifier...")
        self.llm_verifier = UnifiedProductMatchVerifier(
            provider=llm_provider,
            model_name=llm_model,
            temperature=0
        )
        print(f"LLM verifier initialized: {self.llm_verifier.get_info()}")
        
    def add_reference_product(self, name: str, price: float, **metadata):
        """
        Add a reference product to the database.
        
        Args:
            name: Product name
            price: Product price
            **metadata: Additional product information (SKU, category, etc.)
        """
        normalized_name = self._normalize_text(name)
        self.reference_products[normalized_name] = {
            'original_name': name,
            'price': price,
            'metadata': metadata
        }
        
    def load_reference_products_from_list(self, products: List[Dict]):
        """
        Load multiple reference products from a list.
        
        Args:
            products: List of dicts with 'name', 'price', and optional metadata
        """
        for product in products:
            name = product.get('name')
            price = product.get('price')
            metadata = {k: v for k, v in product.items() if k not in ['name', 'price']}
            self.add_reference_product(name, price, **metadata)
            
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for consistent matching."""
        return (text.lower()
                .replace(" ", "")
                .replace("&", "and")
                .replace("+", "plus")
                .replace("-", ""))
    
    @staticmethod
    def _remove_currency_symbols(text: str) -> str:
        """Remove currency symbols from text."""
        return ''.join(
            ch for ch in text 
            if unicodedata.category(ch) != 'Sc'
        )
    
    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _wait_for_rate_limit(self):
        """Ensure minimum delay between API calls to avoid rate limiting."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_delay_between_calls:
            time.sleep(self.min_delay_between_calls - time_since_last_call)
        self.last_api_call_time = time.time()
    
    def _make_api_request_with_retry(self, url: str, payload: dict, headers: dict = None, max_retries: int = 5) -> dict:
        """
        Make API request with retry logic for rate limiting and network errors.
        
        Args:
            url: API endpoint URL
            payload: Request payload
            headers: Optional custom headers (default: Content-Type: application/json)
            max_retries: Maximum number of retry attempts
            
        Returns:
            API response as dict
        """
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                # Longer timeout for large images, with separate connect and read timeouts
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=(10, 60)  # (connect timeout, read timeout)
                )
                
                if response.status_code == 429:
                    # Rate limit hit, wait longer
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except (requests.exceptions.Timeout, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.SSLError) as e:
                wait_time = (2 ** attempt) * 3  # 3, 6, 12, 24, 48 seconds
                if attempt < max_retries - 1:
                    print(f"Network error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Network error after {max_retries} attempts: {e}")
                    raise
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        raise Exception("Max retries exceeded")
    
    def detect_shelf_labels(self, image: np.ndarray, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Detect shelf labels/products using YOLO.
        
        Args:
            image: Input shelf image as numpy array
            confidence_threshold: Minimum confidence score for detections (default: 0.25)
            
        Returns:
            List of detection dicts with 'bbox' and 'confidence'
        """
        # Run YOLO inference
        results = self.yolo_model(image, conf=confidence_threshold, verbose=False)
        
        processed_detections = []
        
        # Process YOLO results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Accept all detections from custom trained model
                processed_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
        
        print(f"YOLO detected {len(processed_detections)} products")
        return processed_detections
    
    def extract_text_from_crop(self, crop: np.ndarray) -> List[Dict]:
        """
        Extract product names and prices from cropped product using vision-capable LLM.
        
        Args:
            crop: Cropped product image
            
        Returns:
            List of dicts with 'item_name', 'price', and visual attributes
        """
        base64_crop = self._encode_image_to_base64(crop)
        
        API_URL = "https://api.openai.com/v1/chat/completions"
        
        prompt_text = """You are an expert at reading product labels and extracting information. Carefully examine this product image.

TASK: Extract the complete product identification information INCLUDING visual characteristics.

EXTRACTION RULES:

1. BRAND NAME (REQUIRED):
   - Main brand (e.g., "Vatika", "Parachute", "Dabur", "Janet", "Kesh King", "Emami")
   - Always include the brand as the first part of the product name

2. PRODUCT LINE/TYPE (REQUIRED):
   - What type of product is it? (e.g., "Hair Oil", "Hair Fall Control", "Herbal Hair Oil", "Natural Hair Oil")
   - Include specific product line names (e.g., "Ayurveda", "Naturals", "Gold", "7 Oils in One")

3. VARIANT/FORMULATION:
   - Key ingredients or benefits (e.g., "Coconut", "Almond", "Hibiscus", "Nourish & Protect")
   - Special formulations (e.g., "7 Oils in One", "Enriched with Henna", "Kalonji", "Shea Butter", "Black Seed")
   - Treatment type (e.g., "Hair Fall Control", "Anti-Dandruff", "Damage Repair")

4. SIZE/VOLUME (if visible):
   - Include measurements (e.g., "200mL", "100ml", "150ml", "300ml")
   - Pack size if mentioned (e.g., "3 pack", "Twin Pack")

5. PRICE (if visible):
   - Extract exact price with currency symbol
   - If no price visible on product itself, leave as empty string ""

6. VISUAL ATTRIBUTES (analyze the product appearance):
   - package_type: "bottle", "box", "jar", "tube", "sachet", "can"
   - primary_color: Dominant color of packaging (e.g., "green", "blue", "yellow", "red", "brown", "gold", "white", "black", "purple", "orange")
   - cap_color: Color of cap/lid if visible (same color options)
   - label_color: Background color of main label (same color options)
   - transparency: "transparent", "opaque", "semi-transparent"

7. SPECIAL NOTES:
   - For hair oils, hair care products: ALWAYS include "Hair Oil" or product type
   - Read text on bottles, boxes, labels carefully
   - Include text in ALL languages if present (English, Hindi, etc.)
   - Maintain proper spacing and capitalization

EXAMPLE OUTPUTS:
✓ {"item_name": "Vatika Enriched Coconut Hair Oil 200ml", "price": "", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "green", "transparency": "transparent"}
✓ {"item_name": "Parachute Advansed Almond Enriched Coconut Hair Oil", "price": "", "package_type": "bottle", "primary_color": "blue", "cap_color": "blue", "label_color": "blue", "transparency": "opaque"}
✓ {"item_name": "Emami 7 Oils in One Hair Oil Shea Butter", "price": "", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "orange", "transparency": "opaque"}

FORMAT: Return as JSON array with objects containing: 'item_name', 'price', 'package_type', 'primary_color', 'cap_color', 'label_color', 'transparency'.
Be precise and extract the FULL product identity, not generic descriptions."""
        
        # Use LangChain-based OCR (supports both OpenAI and Anthropic)
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_crop}"
                    }
                }
            ]
        )
        
        try:
            # Invoke the OCR LLM (access through the verifier's internal LLM)
            response = self.ocr_verifier.verifier.llm.invoke([message])
            content = response.content
            
            # Handle response content - try to extract JSON
            # Some models wrap JSON in markdown code blocks or add extra text
            if isinstance(content, str):
                content = content.strip()
                
                # Remove markdown code blocks if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                    if content.endswith('```'):
                        content = content[:-3]  # Remove closing ```
                elif content.startswith('```'):
                    content = content[3:]  # Remove opening ```
                    if content.endswith('```'):
                        content = content[:-3]  # Remove closing ```
                
                content = content.strip()
                
                # Find JSON array or object boundaries
                # Look for the first [ or { and last ] or }
                start_idx = -1
                end_idx = -1
                
                # Find start of JSON (first [ or {)
                for i, char in enumerate(content):
                    if char in '[{':
                        start_idx = i
                        break
                
                # Find end of JSON (last ] or } that matches)
                if start_idx != -1:
                    bracket_type = content[start_idx]
                    matching_bracket = ']' if bracket_type == '[' else '}'
                    bracket_count = 0
                    
                    for i in range(start_idx, len(content)):
                        if content[i] == bracket_type:
                            bracket_count += 1
                        elif content[i] == matching_bracket:
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i
                                break
                
                # Extract only the JSON portion
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx:end_idx + 1]
            
            # Parse the JSON response
            parsed = json.loads(content)
            
            # Handle both direct array and wrapped object formats
            if isinstance(parsed, list):
                items = parsed
            elif 'items' in parsed:
                items = parsed['items']
            elif 'products' in parsed:
                items = parsed['products']
            else:
                # Try to find the first array in the response
                items = None
                for value in parsed.values():
                    if isinstance(value, list):
                        items = value
                        break
                if items is None:
                    items = [parsed] if 'item_name' in parsed else [{"item_name": "miscellaneous", "price": ""}]
            
            if not items:
                items = [{"item_name": "miscellaneous", "price": ""}]
            
            # Ensure all items have required fields
            for item in items:
                item.setdefault('price', '')
                item.setdefault('package_type', '')
                item.setdefault('primary_color', '')
                item.setdefault('cap_color', '')
                item.setdefault('label_color', '')
                item.setdefault('transparency', '')
                
            return items
        except Exception as e:
            print(f"ERROR extracting text from crop: {e}")
            return [{"item_name": "error_reading_label", "price": ""}]
    
    def match_with_reference(self, extracted_items: List[Dict], fuzzy_threshold: int = 70, llm_verification_threshold: int = 85) -> List[Dict]:
        """
        Match extracted items with reference products using hybrid approach:
        1. Fuzzy matching with visual attributes
        2. LLM verification for ambiguous matches
        
        Args:
            extracted_items: List of extracted products with names and prices
            fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matching (default: 70)
            llm_verification_threshold: Similarity threshold below which LLM verification is used (default: 85)
            
        Returns:
            List of matched products with validation status
        """
        matched_results = []
        
        for item in extracted_items:
            if not item.get('item_name') or item['item_name'].strip() == '':
                # Empty product name
                result = {
                    'item_name': item['item_name'],
                    'extracted_price': item['price'],
                    'status': 'PRODUCT_NOT_FOUND',
                    'reference_price': None,
                    'match': False,
                    'similarity': 0,
                    'match_method': 'none'
                }
                matched_results.append(result)
                continue
            
            normalized_name = self._normalize_text(item['item_name'])
            
            # STEP 1: Try exact match
            reference = self.reference_products.get(normalized_name)
            similarity = 100 if reference else 0
            match_method = 'exact' if reference else None
            
            # STEP 2: Fuzzy matching with visual attribute boosting
            if not reference and self.reference_products:
                ref_names = list(self.reference_products.keys())
                
                # Find best match using fuzzy string matching
                best_match = process.extractOne(
                    normalized_name, 
                    ref_names, 
                    scorer=fuzz.token_sort_ratio
                )
                
                if best_match and best_match[1] >= (fuzzy_threshold - 10):  # Lower threshold for visual boost
                    candidate_ref = self.reference_products[best_match[0]]
                    text_similarity = best_match[1]
                    
                    # Boost similarity score if visual attributes match
                    visual_boost = 0
                    visual_matches = []
                    extracted_attrs = {
                        'package_type': item.get('package_type', ''),
                        'primary_color': item.get('primary_color', ''),
                        'cap_color': item.get('cap_color', ''),
                        'label_color': item.get('label_color', '')
                    }
                    
                    ref_attrs = candidate_ref.get('metadata', {})
                    
                    # Check each visual attribute
                    if extracted_attrs['package_type'] and extracted_attrs['package_type'] == ref_attrs.get('package_type'):
                        visual_boost += 3
                        visual_matches.append('package_type')
                    if extracted_attrs['primary_color'] and extracted_attrs['primary_color'] == ref_attrs.get('primary_color'):
                        visual_boost += 3
                        visual_matches.append('primary_color')
                    if extracted_attrs['cap_color'] and extracted_attrs['cap_color'] == ref_attrs.get('cap_color'):
                        visual_boost += 2
                        visual_matches.append('cap_color')
                    if extracted_attrs['label_color'] and extracted_attrs['label_color'] == ref_attrs.get('label_color'):
                        visual_boost += 2
                        visual_matches.append('label_color')
                    
                    # Apply boost to similarity score
                    boosted_similarity = min(100, text_similarity + visual_boost)
                    
                    if boosted_similarity >= fuzzy_threshold:
                        # STEP 3: LLM verification for ambiguous matches
                        if boosted_similarity < llm_verification_threshold:
                            print(f"  Ambiguous match detected ({boosted_similarity}%), using LLM verification...")
                            # Use LLM to verify the match
                            llm_verified = self._verify_match_with_llm(
                                extracted_item=item,
                                candidate_reference=candidate_ref,
                                text_similarity=boosted_similarity,
                                visual_matches=visual_matches
                            )
                            
                            if llm_verified['is_match']:
                                reference = candidate_ref
                                similarity = llm_verified['confidence']
                                match_method = 'fuzzy+visual+llm'
                            else:
                                # LLM rejected the match
                                reference = None
                                similarity = 0
                                match_method = 'llm_rejected'
                        else:
                            # High confidence fuzzy+visual match, no LLM needed
                            reference = candidate_ref
                            similarity = boosted_similarity
                            match_method = 'fuzzy+visual'
            
            # Build result
            if not reference:
                result = {
                    'item_name': item['item_name'],
                    'extracted_price': item['price'],
                    'status': 'PRODUCT_NOT_FOUND',
                    'reference_price': None,
                    'match': False,
                    'similarity': 0,
                    'match_method': match_method or 'no_match'
                }
            else:
                try:
                    if item['price'] and item['price'].strip():
                        extracted_price = float(self._remove_currency_symbols(item['price']))
                        reference_price = float(reference['price'])
                        
                        if extracted_price == reference_price:
                            status = 'PRICE_MATCH'
                            match = True
                        else:
                            status = 'PRICE_MISMATCH'
                            match = False
                    else:
                        # No price extracted
                        status = 'PRICE_NOT_FOUND'
                        match = False
                        
                except (ValueError, TypeError):
                    status = 'INVALID_PRICE'
                    match = False
                    reference_price = reference['price']
                    
                result = {
                    'item_name': item['item_name'],
                    'extracted_price': item.get('price', ''),
                    'reference_price': reference.get('price'),
                    'status': status,
                    'match': match,
                    'reference_name': reference.get('original_name'),
                    'metadata': reference.get('metadata', {}),
                    'similarity': similarity,
                    'match_method': match_method
                }
                
            matched_results.append(result)
            
        return matched_results
    
    def _verify_match_with_llm(self, extracted_item: Dict, candidate_reference: Dict, text_similarity: float, visual_matches: List[str]) -> Dict:
        """
        Use LangChain-based LLM verifier to validate ambiguous matches.
        Focuses on brand name validation and product type verification.
        
        Args:
            extracted_item: Product extracted from shelf with visual attributes
            candidate_reference: Reference product from database
            text_similarity: Fuzzy text similarity score (0-100)
            visual_matches: List of visual attributes that matched
            
        Returns:
            Dictionary with 'is_match', 'confidence', and 'reasoning'
        """
        # Prepare reference product in expected format
        reference_for_verifier = {
            'name': candidate_reference['original_name'],
            'price': candidate_reference.get('price', 'N/A'),
            'sku': candidate_reference['metadata'].get('sku', 'N/A'),
            'package_type': candidate_reference['metadata'].get('package_type'),
            'primary_color': candidate_reference['metadata'].get('primary_color'),
            'cap_color': candidate_reference['metadata'].get('cap_color'),
            'label_color': candidate_reference['metadata'].get('label_color')
        }
        
        # Use LangChain verifier
        result = self.llm_verifier.verify_match(
            extracted_item=extracted_item,
            candidate_reference=reference_for_verifier,
            text_similarity=text_similarity,
            visual_matches=visual_matches
        )
        
        # Print verification result
        print(f"  🤖 LLM Verification: {'✅ MATCH' if result.get('is_match') else '❌ NO MATCH'} "
              f"(confidence: {result.get('confidence', 0)}%) - {result.get('reasoning', '')}")
        
        return result
    
    def process_shelf_image(self, image_path: str, output_path: str = None) -> Dict:
        """
        Complete pipeline: detect, extract, match, and visualize.
        
        Args:
            image_path: Path to shelf image
            output_path: Optional path to save annotated image
            
        Returns:
            Dict with detection results and matched products
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        print("Detecting shelf labels...")
        detections = self.detect_shelf_labels(image)
        print(f"Found {len(detections)} shelf labels")
        
        all_results = []
        annotated_image = image.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Add offset for better OCR
            offset = 10
            x1 = max(0, x1 - offset)
            y1 = max(0, y1 - offset)
            x2 = min(image.shape[1], x2 + offset)
            y2 = min(image.shape[0], y2 + offset)
            
            # Crop label
            crop = image[y1:y2, x1:x2]
            
            print(f"Processing label {idx+1}/{len(detections)}...")
            extracted_items = self.extract_text_from_crop(crop)
            matched_items = self.match_with_reference(extracted_items)
            
            # Determine overall status for this detection
            if not matched_items:
                status = 'NO_TEXT_FOUND'
                color = (128, 128, 128)  # Gray
            elif all(item['status'] == 'PRICE_MATCH' for item in matched_items):
                status = 'ALL_MATCH'
                color = (0, 255, 0)  # Green
            elif any(item['status'] == 'PRICE_MISMATCH' for item in matched_items):
                status = 'MISMATCH'
                color = (0, 0, 255)  # Red
            elif all(item['status'] == 'PRODUCT_NOT_FOUND' for item in matched_items):
                status = 'NOT_FOUND'
                color = (0, 255, 255)  # Yellow
            else:
                status = 'MIXED'
                color = (255, 0, 255)  # Magenta
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Add SKU label on bounding box if product matched
            if matched_items and matched_items[0].get('metadata'):
                sku = matched_items[0]['metadata'].get('sku', '')
                if sku:
                    # Prepare label text
                    label_text = f"{sku}"
                    
                    # Get text size for background rectangle
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    
                    # Draw background rectangle for text
                    label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
                    cv2.rectangle(annotated_image, 
                                (x1, label_y - text_height - 5), 
                                (x1 + text_width + 5, label_y + 5), 
                                color, -1)
                    
                    # Draw SKU text
                    cv2.putText(annotated_image, label_text, 
                              (x1 + 2, label_y), 
                              font, font_scale, (255, 255, 255), thickness)
            
            all_results.append({
                'bbox': [x1, y1, x2, y2],
                'status': status,
                'items': matched_items
            })
        
        # Save annotated image if requested
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved to {output_path}")
        
        return {
            'total_labels': len(detections),
            'results': all_results,
            'annotated_image': annotated_image
        }


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser for flexible LLM provider selection
    parser = argparse.ArgumentParser(description='Shelf Product Identification System')
    parser.add_argument('--llm-provider', type=str, default='openai', choices=['openai', 'anthropic'],
                        help='LLM provider for match verification (default: openai)')
    parser.add_argument('--llm-model', type=str, default=None,
                        help='Specific LLM model name (optional, uses provider defaults)')
    parser.add_argument('--ocr-provider', type=str, default='openai', choices=['openai', 'anthropic'],
                        help='LLM provider for OCR text extraction (default: openai)')
    parser.add_argument('--ocr-model', type=str, default=None,
                        help='Specific OCR model name (optional, uses provider defaults)')
    parser.add_argument('--image', type=str, default='data/test_images/IMG_2329.jpeg',
                        help='Path to shelf image (default: data/test_images/IMG_2329.jpeg)')
    parser.add_argument('--output', type=str, default='shelf_annotated.jpg',
                        help='Path to save annotated image (default: shelf_annotated.jpg)')
    
    args = parser.parse_args()
    
    # Initialize system - get API key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it using: export OPENAI_API_KEY='your-api-key' "
            "or create a .env file with OPENAI_API_KEY=your-api-key"
        )
    
    # Print configuration
    print("\n" + "="*60)
    print("SHELF PRODUCT IDENTIFICATION SYSTEM")
    print("="*60)
    
    # OCR Configuration
    print(f"OCR Provider: {args.ocr_provider.upper()}")
    if args.ocr_model:
        print(f"OCR Model: {args.ocr_model}")
    else:
        ocr_default_models = {
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-5-haiku-20241022'
        }
        print(f"OCR Model: {ocr_default_models[args.ocr_provider]} (default)")
    
    # Verification Configuration
    print(f"Verification Provider: {args.llm_provider.upper()}")
    if args.llm_model:
        print(f"Verification Model: {args.llm_model}")
    else:
        llm_default_models = {
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-sonnet-4-5-20250929'
        }
        print(f"Verification Model: {llm_default_models[args.llm_provider]} (default)")
    
    print(f"Image: {args.image}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")
    
    # Use your custom trained YOLO model with selected LLM provider
    identifier = ShelfProductIdentifier(
        openai_api_key=OPENAI_API_KEY,
        yolo_model_path='data/best.pt',
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ocr_provider=args.ocr_provider,
        ocr_model=args.ocr_model
    )
    
    # Add reference products for hair care/hair oil shelf
    # Include multiple variants with visual attributes for better fuzzy matching
    reference_products = [
        # Vatika products (typically green bottles)
        {"name": "Vatika Enriched Coconut Hair Oil", "price": 250.00, "sku": "VAT001", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "green"},
        {"name": "Vatika Hibiscus Hair Oil", "price": 180.00, "sku": "VAT002", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "pink"},
        {"name": "Vatika Cactus Hair Oil", "price": 200.00, "sku": "VAT003", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "green"},
        {"name": "Vatika Argan Hair Oil", "price": 220.00, "sku": "VAT004", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "brown"},
        {"name": "Vatika Jasmine Hair Oil", "price": 190.00, "sku": "VAT005", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "white"},
        {"name": "Vatika Castor Hair Oil", "price": 210.00, "sku": "VAT006", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "brown"},
        {"name": "Vatika Almond Hair Oil", "price": 230.00, "sku": "VAT007", "package_type": "bottle", "primary_color": "green", "cap_color": "gold", "label_color": "brown"},
        
        # Parachute products (typically blue/white)
        {"name": "Parachute Advansed Almond Enriched Coconut Hair Oil", "price": 120.00, "sku": "PAR001", "package_type": "bottle", "primary_color": "blue", "cap_color": "blue", "label_color": "blue"},
        {"name": "Parachute Coconut Hair Oil", "price": 100.00, "sku": "PAR002", "package_type": "bottle", "primary_color": "blue", "cap_color": "blue", "label_color": "blue"},
        {"name": "Parachute Ayurvedic Hair Oil", "price": 130.00, "sku": "PAR003", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        
        # Dabur products (typically brown/red)
        {"name": "Dabur Amla Hair Oil", "price": 150.00, "sku": "DAB001", "package_type": "bottle", "primary_color": "brown", "cap_color": "red", "label_color": "red"},
        
        # Kesh King products (brown/gold)
        {"name": "Kesh King Ayurvedic Scalp and Hair Oil", "price": 280.00, "sku": "KES001", "package_type": "bottle", "primary_color": "brown", "cap_color": "gold", "label_color": "gold"},
        {"name": "Kesh King Ayurvedic Oil", "price": 280.00, "sku": "KES002", "package_type": "bottle", "primary_color": "brown", "cap_color": "gold", "label_color": "gold"},
        
        # Janet Ayurveda products (various colors by variant)
        {"name": "Janet Ayurveda Hair Fall Control", "price": 220.00, "sku": "JAN001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Janet Ayurveda Dark Henna Hair Oil", "price": 230.00, "sku": "JAN002", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "brown"},
        {"name": "Janet Ayurveda Dandruff Control Indigo Herb Hair Oil", "price": 240.00, "sku": "JAN003", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "blue"},
        {"name": "Janet Ayurveda Dry Hair Treatment Neeli Gotukola Hair Oil", "price": 235.00, "sku": "JAN004", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "purple"},
        {"name": "Janet Ayurveda Nelli Gotukola Hair Oil", "price": 225.00, "sku": "JAN005", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        
        # Kashvi products
        {"name": "Kashvi Herbal Hair Oil", "price": 95.00, "sku": "KAS001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        
        # Navratna products (typically green)
        {"name": "Navratna Cool Hair Oil", "price": 85.00, "sku": "NAV001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Navratna Herbal Hair Oil", "price": 85.00, "sku": "NAV002", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Navratna Herbal Oil", "price": 85.00, "sku": "NAV003", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        
        # Emami 7 Oils products (with visual attributes)
        {"name": "Emami 7 Oils in One Hair Oil", "price": 175.00, "sku": "EMA001", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "gold"},
        {"name": "Emami 7 Oils in One Non Sticky Hair Oil", "price": 180.00, "sku": "EMA002", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "green"},
        {"name": "7 Oils in One Hair Oil Shea Butter", "price": 185.00, "sku": "EMA003", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "orange"},
        {"name": "7 Oils in One Hair Oil Black Seed", "price": 190.00, "sku": "EMA004", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "black"},
        {"name": "7 Oils in One Hair Oil Aloe Vera", "price": 175.00, "sku": "EMA005", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "green"},
        
        # Kumarika products (green bottles)
        {"name": "Kumarika Herbal Hair Oil", "price": 120.00, "sku": "KUM001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Kumarika Coconut Hair Oil", "price": 110.00, "sku": "KUM002", "package_type": "bottle", "primary_color": "green", "cap_color": "white", "label_color": "green"},
        
        # Other brands with visual attributes
        {"name": "Bajaj Almond Drops Hair Oil", "price": 110.00, "sku": "BAJ001", "package_type": "bottle", "primary_color": "red", "cap_color": "red", "label_color": "red"},
        {"name": "Sliming Herbal Hair Treatment", "price": 160.00, "sku": "SLI001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Ganesha Hair Oil", "price": 140.00, "sku": "GEN001", "package_type": "bottle", "primary_color": "brown", "cap_color": "gold", "label_color": "gold"},
        {"name": "Disaar Argan Oil", "price": 250.00, "sku": "DIS001", "package_type": "bottle", "primary_color": "white", "cap_color": "gold", "label_color": "gold"},
        {"name": "Mega Extra Virgin Olive Oil", "price": 300.00, "sku": "MEG001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Dinuji Herbal Castor Oil", "price": 150.00, "sku": "DIN001", "package_type": "bottle", "primary_color": "brown", "cap_color": "brown", "label_color": "brown"},
        {"name": "Shell Vitamin E Hair Oil", "price": 130.00, "sku": "SHE001", "package_type": "bottle", "primary_color": "purple", "cap_color": "purple", "label_color": "purple"},
        {"name": "Shahnaz Husain Vitamin E Hair Oil", "price": 200.00, "sku": "SHA001", "package_type": "bottle", "primary_color": "purple", "cap_color": "gold", "label_color": "purple"},
        {"name": "Good Look Coconut Hair Oil", "price": 90.00, "sku": "GOO001", "package_type": "bottle", "primary_color": "yellow", "cap_color": "yellow", "label_color": "yellow"},
        {"name": "Golden Touch Hair Oil", "price": 125.00, "sku": "GOL001", "package_type": "bottle", "primary_color": "gold", "cap_color": "gold", "label_color": "gold"},
        {"name": "Rishi Coconut Hair Oil", "price": 95.00, "sku": "RIS001", "package_type": "bottle", "primary_color": "blue", "cap_color": "blue", "label_color": "blue"},
        {"name": "Liz Herbal Hair Oil", "price": 105.00, "sku": "LIZ001", "package_type": "bottle", "primary_color": "green", "cap_color": "green", "label_color": "green"},
        {"name": "Kutimaya Hair Oil", "price": 115.00, "sku": "KUT001", "package_type": "bottle", "primary_color": "brown", "cap_color": "brown", "label_color": "brown"},
    ]
    
    identifier.load_reference_products_from_list(reference_products)
    
    # Process shelf image
    results = identifier.process_shelf_image(
        image_path=args.image,
        output_path=args.output
    )
    
    # Print results
    print("\n" + "="*60)
    print("SHELF PRODUCT IDENTIFICATION RESULTS")
    print("="*60)
    print(f"Total products detected: {results['total_labels']}")
    print("="*60)
    
    matches = 0
    mismatches = 0
    not_found = 0
    price_not_found = 0
    
    for idx, result in enumerate(results['results']):
        print(f"\n[Product {idx+1}] Status: {result['status']}")
        print("-" * 60)
        for item in result['items']:
            print(f"  Product: {item['item_name']}")
            print(f"  Extracted Price: {item['extracted_price']}")
            if item.get('reference_price'):
                print(f"  Reference Price: {item['reference_price']}")
                print(f"  Reference Name: {item.get('reference_name', 'N/A')}")
                print(f"  Match Similarity: {item.get('similarity', 0)}%")
                print(f"  Match Method: {item.get('match_method', 'unknown')}")
            print(f"  Validation: {item['status']}")
            
            if item['status'] == 'PRICE_MATCH':
                matches += 1
            elif item['status'] == 'PRICE_MISMATCH':
                mismatches += 1
            elif item['status'] == 'PRODUCT_NOT_FOUND':
                not_found += 1
            elif item['status'] == 'PRICE_NOT_FOUND':
                price_not_found += 1
        print("-" * 60)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Price Matches: {matches}")
    print(f"✗ Price Mismatches: {mismatches}")
    print(f"⚠ Price Not Found in Image: {price_not_found}")
    print(f"? Products Not Found in Reference: {not_found}")
    print("="*60)