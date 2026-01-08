from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import os
import sys
from typing import Literal, Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from rapidfuzz import fuzz
from src.exception import MerchandiserException
from src.logs import logging

# Load environment variables
_ = load_dotenv()

LLMProvider = Literal["openai", "anthropic"]

class MatchResult(BaseModel):
    """Pydantic model for structured match result output."""
    is_match: bool = Field(description="Whether a match was found in the reference database")
    matched_product_id: str = Field(description="ID of the matched product from reference database, or 'None' if no match")
    confidence_score: float = Field(description="Confidence score of the match (0.0 to 1.0)")
    reasoning: str = Field(description="Explanation of why this product was matched or not matched")


class LLMProductMatcher:
    """
    A class to match products using LLM.
    """
    def __init__(
        self, reference_db_path: str = 'data/product_reference_database.json',
        llm_provider: LLMProvider = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
        top_k_candidates: int = 10,
        similarity_threshold: int = 50,
        confidence_threshold: float = 0.75
    ):
        self.reference_db_path = Path(reference_db_path)
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_k_candidates = top_k_candidates
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Set default models if not specified
        if model_name is None:
            if llm_provider == "openai":
                model_name = "gpt-4o-mini"
            elif llm_provider == "anthropic":
                model_name = "claude-sonnet-4-5-20250929"
        
        self.model_name = model_name
        
        # Get API key
        if api_key is None:
            if llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif llm_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if api_key is None:
            error_msg = f"{llm_provider.upper()} API key not found. Please set it in .env file or pass as parameter."
            logging.error(error_msg)
            raise MerchandiserException(error_msg, sys)
        
        # Initialize LLM
        try:
            if llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    api_key=api_key,
                    temperature=temperature
                )
            elif llm_provider == "anthropic":
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    api_key=api_key,
                    temperature=temperature
                )
            
            # Initialize JSON parser
            self.parser = JsonOutputParser(pydantic_object=MatchResult)
            logging.info(f"LLMMatcher initialized with {llm_provider.upper()} model: {model_name}")
        
        except Exception as e:
            logging.error(f"Error initializing LLMMatcher: {e}")
            raise MerchandiserException(f"Error initializing LLMMatcher: {e}", sys) from e
        
        # Load reference database
        try:
            self.reference_products = self._load_reference_database()
            logging.info(f"Loaded {len(self.reference_products)} reference products from {self.reference_db_path}")
        except Exception as e:
            logging.error(f"Error loading reference database: {e}")
            raise MerchandiserException(f"Error loading reference database: {e}", sys) from e
        
    def _load_reference_database(self):
        """
        Load the reference database from the file.
        """
        if not self.reference_db_path.exists():
            logging.warning(f"Reference database not found at {self.reference_db_path}")
            return {}
        try:
            with open(self.reference_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading reference database: {e}")
            raise MerchandiserException(f"Error loading reference database: {e}", sys) from e
        
    def _calculate_similarity(self, detected: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """
        Calculate composite similarity score using multiple attributes with weighted scoring.
        """
        scores = []
        weights = []
        
        # Extract product data if nested
        detected_product = detected.get('product', detected) if isinstance(detected.get('product'), dict) else detected
        reference_product = reference.get('product', reference) if isinstance(reference.get('product'), dict) else reference
        
        # Brand matching (weight: 0.35)
        detected_brand = detected_product.get('brand', '').lower().strip()
        reference_brand = reference_product.get('brand', '').lower().strip()
        if detected_brand and reference_brand:
            brand_score = fuzz.token_sort_ratio(detected_brand, reference_brand)
            scores.append(brand_score)
            weights.append(0.35)
        
        # Product name matching (weight: 0.35)
        detected_name = detected_product.get('product_name', '').lower().strip()
        reference_name = reference_product.get('product_name', '').lower().strip()
        if detected_name and reference_name:
            name_score = fuzz.token_sort_ratio(detected_name, reference_name)
            scores.append(name_score)
            weights.append(0.35)
        
        # Variant matching (weight: 0.10)
        detected_variant = detected_product.get('variant', '').lower().strip()
        reference_variant = reference_product.get('variant', '').lower().strip()
        if detected_variant and reference_variant:
            variant_score = fuzz.token_sort_ratio(detected_variant, reference_variant)
            scores.append(variant_score)
            weights.append(0.10)
        
        # Color matching (weight: 0.10)
        detected_colors = [c.lower().strip() for c in detected_product.get('primary_colors', []) if c]
        reference_colors = [c.lower().strip() for c in reference_product.get('primary_colors', []) if c]
        if detected_colors and reference_colors:
            color_overlap = len(set(detected_colors) & set(reference_colors))
            color_score = (color_overlap / max(len(detected_colors), len(reference_colors))) * 100
            scores.append(color_score)
            weights.append(0.10)
        
        # Packaging type matching (weight: 0.05)
        detected_packaging = detected_product.get('packaging_type', '').lower().strip()
        reference_packaging = reference_product.get('packaging_type', '').lower().strip()
        if detected_packaging and reference_packaging:
            packaging_score = fuzz.ratio(detected_packaging, reference_packaging)
            scores.append(packaging_score)
            weights.append(0.05)
        
        # Size/volume matching (weight: 0.05)
        detected_size = detected_product.get('size_volume', '').lower().strip()
        reference_size = reference_product.get('size_volume', '').lower().strip()
        if detected_size and reference_size and detected_size != 'unknown' and reference_size != 'unknown':
            size_score = fuzz.partial_ratio(detected_size, reference_size)
            scores.append(size_score)
            weights.append(0.05)
        
        # Calculate weighted average
        if not scores:
            return 0.0
        
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return round(weighted_score, 2)
    
    def _filter_candidate_products(self, detected_product_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter top-k candidate products using RapidFuzz similarity matching.
        
        Args:
            detected_product_info: Extracted product information dictionary
            
        Returns:
            List of top candidates with similarity scores
        """
        candidates = []
        
        for product_id, product_data in self.reference_products.items():
            similarity_score = self._calculate_similarity(
                detected_product_info,
                product_data
            )
            
            # Only consider candidates above threshold
            if similarity_score >= self.similarity_threshold:
                candidates.append({
                    'product_id': str(product_id),
                    'product_data': product_data,
                    'similarity_score': similarity_score
                })
        
        # Sort by similarity score (descending) and take top-k
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_candidates = candidates[:self.top_k_candidates]
        
        logging.info(
            f"Filtered {len(candidates)} candidates above threshold ({self.similarity_threshold}%), "
            f"selected top {len(top_candidates)} for LLM verification"
        )
        
        return top_candidates
    def match(self, detected_product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match detected product using two-stage approach:
        1. RapidFuzz filtering to get top-k candidates
        2. LLM verification for final match decision
        """
        try:
            # Stage 1: Fast filtering with RapidFuzz
            candidates = self._filter_candidate_products(detected_product_info)
            
            if not candidates:
                return {
                    'is_match': False,
                    'matched_product_id': 'None',
                    'confidence_score': 0.0,
                    'reasoning': f'No candidate products found with similarity >= {self.similarity_threshold}%'
                }
            
            # If only one high-confidence candidate, skip LLM
            if len(candidates) == 1 and candidates[0]['similarity_score'] > 90:
                logging.info(
                    f"High confidence match found: {candidates[0]['product_id']} "
                    f"({candidates[0]['similarity_score']:.1f}%) - LLM verification skipped"
                )
                return {
                    'is_match': True,
                    'matched_product_id': candidates[0]['product_id'],
                    'confidence_score': candidates[0]['similarity_score'] / 100.0,
                    'reasoning': f"High similarity match ({candidates[0]['similarity_score']:.1f}%) - LLM verification skipped"
                }
            
            # Stage 2: LLM verification for ambiguous cases
            logging.info(f"Sending {len(candidates)} candidates to LLM for final verification...")
            
            filtered_ref_db = {
                c['product_id']: c['product_data']
                for c in candidates
            }
            
            prompt = f"""
            You are a product matching expert. Compare the detected product with pre-filtered candidate products.
            
            **Detected Product Information:**
            {json.dumps(detected_product_info, indent=2, ensure_ascii=False)}
            
            **Top Candidate Products (pre-filtered by similarity scores):**
            {json.dumps(filtered_ref_db, indent=2, ensure_ascii=False)}
            
            **Pre-filter Similarity Scores:**
            {json.dumps({c['product_id']: f"{c['similarity_score']:.1f}%" for c in candidates}, indent=2)}
            
            **Instructions:**
            1. These candidates were pre-selected based on brand, name, color, and packaging similarity
            2. Compare ALL product attributes: brand, product_name, variant, colors, packaging, size, distinctive elements
            3. Consider that product names may vary slightly (abbreviations, word order, etc.)
            4. If a confident match exists (>75% certainty), set is_match=true with the product_id
            5. If candidates are too different or ambiguous, set is_match=false
            6. Provide confidence_score (0.0-1.0) and detailed reasoning
            
            {self.parser.get_format_instructions()}
            """
            
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            result = self.parser.parse(response.content)
            
            logging.info(
                f"LLM Decision: {'MATCH' if result['is_match'] else 'NO MATCH'} - "
                f"Product ID: {result['matched_product_id']} - "
                f"Confidence: {result['confidence_score']:.2f}"
            )
            
            return result
        
        except Exception as e:
            logging.error(f"Error matching product: {e}")
            raise MerchandiserException(f"Error matching product: {e}", sys) from e

    def batch_match(self, detected_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match multiple detected products efficiently.
        """
        results = []
        logging.info(f"Starting batch matching for {len(detected_products)} products")
        
        for i, product in enumerate(detected_products, 1):
            logging.info(f"Matching product {i}/{len(detected_products)}")
            match_result = self.match(product)
            results.append({
                'detected_product': product,
                'match_result': match_result
            })
        
        # Summary
        matched = sum(1 for r in results if r['match_result']['is_match'])
        logging.info(
            f"Batch matching complete: {matched}/{len(results)} products matched "
            f"({matched/len(results)*100:.1f}% match rate)"
        )
        
        return results