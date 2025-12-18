"""
Unified LLM Verifier that supports multiple providers (OpenAI, Anthropic).
Allows easy switching between different LLM models for product match verification.
"""

import os
from typing import Dict, Any, List, Literal
from dotenv import load_dotenv
from llm_verifier import ProductMatchVerifier
from llm_verifier_anthropic import AnthropicProductMatchVerifier

# Load environment variables
load_dotenv()

LLMProvider = Literal["openai", "anthropic"]


class UnifiedProductMatchVerifier:
    """
    Unified interface for product match verification across multiple LLM providers.
    Supports OpenAI (GPT-4) and Anthropic (Claude) models.
    """
    
    def __init__(
        self,
        provider: LLMProvider = "openai",
        model_name: str = None,
        temperature: float = 0
    ):
        """
        Initialize the unified LLM verifier.
        
        Args:
            provider: LLM provider to use ("openai" or "anthropic")
            model_name: Specific model name (optional, uses defaults if not provided)
                       OpenAI defaults: "gpt-4o-mini" (cost-effective) or "gpt-4o" (most capable)
                       Anthropic defaults: "claude-sonnet-4-5-20250929" (latest, best) or "claude-3-5-haiku-20241022" (fastest)
            temperature: Model temperature (0 for deterministic responses)
        """
        self.provider = provider
        self.temperature = temperature
        
        # Set default models if not specified
        if model_name is None:
            if provider == "openai":
                model_name = "gpt-4o-mini"  # Cost-effective default
            elif provider == "anthropic":
                model_name = "claude-sonnet-4-5-20250929"  # Best balance
        
        self.model_name = model_name
        
        # Initialize the appropriate verifier
        if provider == "openai":
            print(f"🤖 Initializing OpenAI verifier with model: {model_name}")
            self.verifier = ProductMatchVerifier(
                model_name=model_name,
                temperature=temperature
            )
        elif provider == "anthropic":
            print(f"🤖 Initializing Anthropic verifier with model: {model_name}")
            self.verifier = AnthropicProductMatchVerifier(
                model_name=model_name,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'anthropic'.")
    
    def verify_match(
        self,
        extracted_item: Dict[str, Any],
        candidate_reference: Dict[str, Any],
        text_similarity: float,
        visual_matches: List[str]
    ) -> Dict[str, Any]:
        """
        Verify if an extracted product matches a reference product.
        Delegates to the appropriate provider's verifier.
        
        Args:
            extracted_item: Product extracted from shelf image with visual attributes
            candidate_reference: Reference product from database
            text_similarity: Fuzzy text similarity score (0-100)
            visual_matches: List of visual attributes that matched
        
        Returns:
            Dictionary with 'is_match', 'confidence', and 'reasoning'
        """
        return self.verifier.verify_match(
            extracted_item=extracted_item,
            candidate_reference=candidate_reference,
            text_similarity=text_similarity,
            visual_matches=visual_matches
        )
    
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
        return self.verifier.batch_verify_matches(matches_to_verify)
    
    def get_info(self) -> Dict[str, str]:
        """Get information about the current verifier configuration."""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "temperature": self.temperature
        }


# Example usage and comparison
if __name__ == "__main__":
    # Test data
    extracted = {
        'item_name': 'Emami 7 Oils Hair Oil',
        'price': 'Rs. 45',
        'package_type': 'bottle',
        'primary_color': 'green',
        'cap_color': 'red'
    }
    
    reference_wrong = {
        'name': 'Emrad 7 Oils Hair Oil',
        'price': 'Rs. 45',
        'sku': 'EMR-7OIL-45',
        'package_type': 'bottle',
        'primary_color': 'green',
        'cap_color': 'red'
    }
    
    reference_correct = {
        'name': 'Emami 7 Oils in One Hair Oil',
        'price': 'Rs. 45',
        'sku': 'EMA-7OIL-45',
        'package_type': 'bottle',
        'primary_color': 'green',
        'cap_color': 'red'
    }
    
    # Test with OpenAI
    print("\n" + "="*60)
    print("TESTING WITH OPENAI GPT-4o-mini")
    print("="*60)
    
    openai_verifier = UnifiedProductMatchVerifier(provider="openai", model_name="gpt-4o-mini")
    
    print("\n--- Test 1: Different Brand (Expected: NO MATCH) ---")
    result1 = openai_verifier.verify_match(
        extracted_item=extracted,
        candidate_reference=reference_wrong,
        text_similarity=85.0,
        visual_matches=['package_type', 'primary_color', 'cap_color']
    )
    print(f"✓ Match: {result1['is_match']} | Confidence: {result1['confidence']}%")
    print(f"✓ Reasoning: {result1['reasoning']}")
    
    print("\n--- Test 2: Same Brand (Expected: MATCH) ---")
    result2 = openai_verifier.verify_match(
        extracted_item=extracted,
        candidate_reference=reference_correct,
        text_similarity=92.0,
        visual_matches=['package_type', 'primary_color', 'cap_color']
    )
    print(f"✓ Match: {result2['is_match']} | Confidence: {result2['confidence']}%")
    print(f"✓ Reasoning: {result2['reasoning']}")
    
    # Test with Anthropic (if API key available)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n" + "="*60)
        print("TESTING WITH ANTHROPIC CLAUDE")
        print("="*60)
        
        anthropic_verifier = UnifiedProductMatchVerifier(
            provider="anthropic",
            model_name="claude-sonnet-4-5-20250929"
        )
        
        print("\n--- Test 1: Different Brand (Expected: NO MATCH) ---")
        result3 = anthropic_verifier.verify_match(
            extracted_item=extracted,
            candidate_reference=reference_wrong,
            text_similarity=85.0,
            visual_matches=['package_type', 'primary_color', 'cap_color']
        )
        print(f"✓ Match: {result3['is_match']} | Confidence: {result3['confidence']}%")
        print(f"✓ Reasoning: {result3['reasoning']}")
        
        print("\n--- Test 2: Same Brand (Expected: MATCH) ---")
        result4 = anthropic_verifier.verify_match(
            extracted_item=extracted,
            candidate_reference=reference_correct,
            text_similarity=92.0,
            visual_matches=['package_type', 'primary_color', 'cap_color']
        )
        print(f"✓ Match: {result4['is_match']} | Confidence: {result4['confidence']}%")
        print(f"✓ Reasoning: {result4['reasoning']}")
    else:
        print("\n⚠️ ANTHROPIC_API_KEY not found in .env file. Skipping Anthropic tests.")
        print("Add ANTHROPIC_API_KEY to .env to test Claude models.")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
