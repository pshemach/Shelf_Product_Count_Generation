"""
LLM-based product matching verification using Anthropic Claude via LangChain.
This module provides functionality to verify ambiguous product matches using Claude models.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()


class MatchVerificationResult(BaseModel):
    """Schema for LLM match verification response."""
    is_match: bool = Field(description="Whether the extracted product matches the reference product")
    confidence: int = Field(description="Confidence score from 0-100")
    reasoning: str = Field(description="Explanation for the decision")


class AnthropicProductMatchVerifier:
    """
    Verifies ambiguous product matches using LangChain and Anthropic Claude.
    Uses structured output parsing for reliable JSON responses.
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929", temperature: float = 0):
        """
        Initialize the Anthropic LLM verifier.
        
        Args:
            model_name: Anthropic model to use (default: claude-sonnet-4-5-20250929)
                       Options: claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
            temperature: Model temperature (0 for deterministic responses)
        """
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024
        )
        self.parser = JsonOutputParser(pydantic_object=MatchVerificationResult)
    
    def verify_match(
        self,
        extracted_item: Dict[str, Any],
        candidate_reference: Dict[str, Any],
        text_similarity: float,
        visual_matches: List[str]
    ) -> Dict[str, Any]:
        """
        Verify if an extracted product matches a reference product using Claude.
        
        Args:
            extracted_item: Product extracted from shelf image with visual attributes
            candidate_reference: Reference product from database
            text_similarity: Fuzzy text similarity score (0-100)
            visual_matches: List of visual attributes that matched
        
        Returns:
            Dictionary with 'is_match', 'confidence', and 'reasoning'
        """
        # Build the verification prompt
        system_message = SystemMessage(content="""You are an expert product matching system for retail shelf analysis.
Your task is to verify if a product extracted from a shelf image matches a reference product in the database.

CRITICAL EVALUATION CRITERIA (in order of importance):
1. **Brand Name** - MOST IMPORTANT: Brand names must match exactly or be clear variations (e.g., "Vatika" vs "Dabur Vatika").
   - REJECT if brands are completely different (e.g., "Emami" ≠ "Emrad", "Parachute" ≠ "Parachutex")
2. **Product Type** - Product category should match (e.g., both are hair oils, both are shampoos)
3. **Variant Details** - Additional descriptors can vary (e.g., "Coconut" vs "Natural Coconut" is OK)
4. **Visual Attributes** - Colors and packaging provide supporting evidence but are NOT definitive
   - Same colors increase confidence but different colors don't automatically mean mismatch (lighting variations)
   - Package type should generally match

DECISION GUIDELINES:
- HIGH CONFIDENCE MATCH (90-100%): Brand and product type match perfectly, visual attributes support the match
- MEDIUM CONFIDENCE MATCH (70-89%): Brand and product type match, some visual discrepancies acceptable
- LOW CONFIDENCE MATCH (50-69%): Likely the same product but with significant uncertainty
- NO MATCH (0-49%): Different brands, different product types, or clear contradictions

EXAMPLES:
✅ MATCH: "Emami 7 Oils" → "Emami 7 Oils in One" (same brand, same product)
✅ MATCH: "Dabur Vatika Coconut" → "Vatika Enriched Coconut Hair Oil" (same brand family, same product)
✅ MATCH: "Parachute Coconut Oil 200ml" → "Parachute Coconut Hair Oil" (same brand, same product, size variation)
❌ NO MATCH: "Emami Hair Oil" → "Emrad Hair Oil" (different brands)
❌ NO MATCH: "Parachute" → "Parachutex" (different brands)
❌ NO MATCH: "Pantene Shampoo" → "Pantene Conditioner" (different product types)

Return your response in the following JSON format:
{
    "is_match": true/false,
    "confidence": 0-100,
    "reasoning": "Brief explanation of your decision"
}""")
        
        # Build visual attributes strings
        extracted_visual = []
        if extracted_item.get('package_type'):
            extracted_visual.append(f"Package: {extracted_item['package_type']}")
        if extracted_item.get('primary_color'):
            extracted_visual.append(f"Primary color: {extracted_item['primary_color']}")
        if extracted_item.get('cap_color'):
            extracted_visual.append(f"Cap color: {extracted_item['cap_color']}")
        if extracted_item.get('label_color'):
            extracted_visual.append(f"Label color: {extracted_item['label_color']}")
        
        reference_visual = []
        if candidate_reference.get('package_type'):
            reference_visual.append(f"Package: {candidate_reference['package_type']}")
        if candidate_reference.get('primary_color'):
            reference_visual.append(f"Primary color: {candidate_reference['primary_color']}")
        if candidate_reference.get('cap_color'):
            reference_visual.append(f"Cap color: {candidate_reference['cap_color']}")
        if candidate_reference.get('label_color'):
            reference_visual.append(f"Label color: {candidate_reference['label_color']}")
        
        # Construct the user message
        user_message = HumanMessage(content=f"""Please verify this product match:

EXTRACTED FROM SHELF:
- Product Name: "{extracted_item.get('item_name', 'Unknown')}"
- Price: {extracted_item.get('price', 'N/A')}
- Visual Attributes: {', '.join(extracted_visual) if extracted_visual else 'None extracted'}

REFERENCE PRODUCT:
- Product Name: "{candidate_reference.get('name', 'Unknown')}"
- Price: {candidate_reference.get('price', 'N/A')}
- SKU: {candidate_reference.get('sku', 'N/A')}
- Visual Attributes: {', '.join(reference_visual) if reference_visual else 'None available'}

MATCHING INFORMATION:
- Text Similarity: {text_similarity:.1f}%
- Visual Matches: {', '.join(visual_matches) if visual_matches else 'None'}

Based on the criteria above, is this a correct match? Return ONLY valid JSON.""")
        
        try:
            # Invoke Claude with structured output
            response = self.llm.invoke([system_message, user_message])
            
            # Parse the JSON response - returns a dict
            result = self.parser.parse(response.content)
            
            # Result is already a dict from the parser
            return {
                'is_match': result.get('is_match', False),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            print(f"  ⚠️ Anthropic LLM verification error: {str(e)}")
            # Fallback: use text similarity threshold
            fallback_match = text_similarity >= 75
            return {
                'is_match': fallback_match,
                'confidence': int(text_similarity),
                'reasoning': f"Fallback decision due to LLM error. Text similarity: {text_similarity:.1f}%"
            }
    
    def batch_verify_matches(
        self,
        matches_to_verify: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verify multiple matches in batch.
        
        Args:
            matches_to_verify: List of dicts containing extracted_item, candidate_reference, 
                             text_similarity, and visual_matches
        
        Returns:
            List of verification results
        """
        results = []
        for match_data in matches_to_verify:
            result = self.verify_match(
                extracted_item=match_data['extracted_item'],
                candidate_reference=match_data['candidate_reference'],
                text_similarity=match_data['text_similarity'],
                visual_matches=match_data['visual_matches']
            )
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize Anthropic verifier
    verifier = AnthropicProductMatchVerifier(model_name="claude-sonnet-4-5-20250929")
    
    # Test case 1: Similar names but different brands (should reject)
    print("\n=== Test 1: Different Brands (Expected: NO MATCH) ===")
    extracted = {
        'item_name': 'Emami 7 Oils Hair Oil',
        'price': 'Rs. 45',
        'package_type': 'bottle',
        'primary_color': 'green',
        'cap_color': 'red'
    }
    
    reference = {
        'name': 'Emrad 7 Oils Hair Oil',
        'price': 'Rs. 45',
        'sku': 'EMR-7OIL-45',
        'package_type': 'bottle',
        'primary_color': 'green',
        'cap_color': 'red'
    }
    
    result = verifier.verify_match(
        extracted_item=extracted,
        candidate_reference=reference,
        text_similarity=85.0,
        visual_matches=['package_type', 'primary_color', 'cap_color']
    )
    
    print(f"Match: {result['is_match']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Reasoning: {result['reasoning']}")
    
    # Test case 2: Same brand with variant (should match)
    print("\n=== Test 2: Same Brand, Similar Variant (Expected: MATCH) ===")
    extracted2 = {
        'item_name': 'Parachute Coconut Oil 200ml',
        'price': 'Rs. 65',
        'package_type': 'bottle',
        'primary_color': 'blue',
        'cap_color': 'white'
    }
    
    reference2 = {
        'name': 'Parachute Coconut Hair Oil',
        'price': 'Rs. 65',
        'sku': 'PAR-COCO-65',
        'package_type': 'bottle',
        'primary_color': 'blue',
        'cap_color': 'white'
    }
    
    result2 = verifier.verify_match(
        extracted_item=extracted2,
        candidate_reference=reference2,
        text_similarity=78.0,
        visual_matches=['package_type', 'primary_color', 'cap_color']
    )
    
    print(f"Match: {result2['is_match']}")
    print(f"Confidence: {result2['confidence']}%")
    print(f"Reasoning: {result2['reasoning']}")
