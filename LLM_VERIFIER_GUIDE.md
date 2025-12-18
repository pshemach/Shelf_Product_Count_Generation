# LLM Verifier - Multi-Provider Support

This project supports multiple LLM providers for product match verification. You can easily switch between OpenAI (GPT-4) and Anthropic (Claude) models.

## Supported Providers

### 1. OpenAI (GPT-4)

- **File**: `llm_verifier.py`
- **Models**:
  - `gpt-4o-mini` (default, cost-effective)
  - `gpt-4o` (most capable)
- **API Key**: Set `OPENAI_API_KEY` in `.env`

### 2. Anthropic (Claude)

- **File**: `llm_verifier_anthropic.py`
- **Models**:
  - `claude-sonnet-4-5-20250929` (default, latest and most capable)
  - `claude-3-5-sonnet-20241022` (previous generation, best balance)
  - `claude-3-5-haiku-20241022` (fastest, most cost-effective)
- **API Key**: Set `ANTHROPIC_API_KEY` in `.env`

### 3. Unified Interface

- **File**: `llm_verifier_unified.py`
- **Purpose**: Single interface to switch between providers easily

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Add your API keys to `.env` file:

```env
# OpenAI (Required for OCR)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Optional, only if using Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Other keys
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=Product Count
GIMINI_API_KEY=your_gimini_api_key_here
```

## Usage Examples

### Using OpenAI (Default)

```python
from test import ShelfProductIdentifier

# Initialize with OpenAI (default)
identifier = ShelfProductIdentifier(
    openai_api_key="your_key",
    yolo_model_path="data/best.pt",
    llm_provider="openai",  # Default
    llm_model="gpt-4o-mini"  # Optional, uses default if not specified
)

# Process image
results = identifier.process_shelf_image(
    image_path="data/test_images/image.png",
    output_path="output_annotated.jpg"
)
```

### Using Anthropic Claude

```python
from test import ShelfProductIdentifier

# Initialize with Anthropic Claude
identifier = ShelfProductIdentifier(
    openai_api_key="your_openai_key",  # Still needed for OCR
    yolo_model_path="data/best.pt",
    llm_provider="anthropic",
    llm_model="claude-sonnet-4-5-20250929"  # Latest Claude Sonnet 4.5
)

# Process image
results = identifier.process_shelf_image(
    image_path="data/test_images/image.png",
    output_path="output_annotated.jpg"
)
```

### Testing Individual Verifiers

#### Test OpenAI Verifier

```bash
python llm_verifier.py
```

#### Test Anthropic Verifier

```bash
python llm_verifier_anthropic.py
```

#### Test Both and Compare

```bash
python llm_verifier_unified.py
```

## Standalone Usage

You can also use the verifiers independently:

### OpenAI Verifier

```python
from llm_verifier import ProductMatchVerifier

verifier = ProductMatchVerifier(model_name="gpt-4o-mini")

result = verifier.verify_match(
    extracted_item={
        'item_name': 'Emami 7 Oils Hair Oil',
        'package_type': 'bottle',
        'primary_color': 'green'
    },
    candidate_reference={
        'name': 'Emami 7 Oils in One Hair Oil',
        'sku': 'EMA-7OIL-45',
        'package_type': 'bottle'
    },
    text_similarity=85.0,
    visual_matches=['package_type']
)

print(f"Match: {result['is_match']}")
print(f"Confidence: {result['confidence']}%")
print(f"Reasoning: {result['reasoning']}")
```

### Anthropic Verifier

```python
from llm_verifier_anthropic import AnthropicProductMatchVerifier

verifier = AnthropicProductMatchVerifier(
    model_name="claude-sonnet-4-5-20250929"  # Latest Claude Sonnet 4.5
)

result = verifier.verify_match(
    extracted_item={...},
    candidate_reference={...},
    text_similarity=85.0,
    visual_matches=['package_type']
)
```

## Cost Comparison

| Provider  | Model                   | Cost (per 1M tokens)         | Best For                    |
| --------- | ----------------------- | ---------------------------- | --------------------------- |
| OpenAI    | gpt-4o-mini             | Input: $0.15, Output: $0.60  | Cost-effective verification |
| OpenAI    | gpt-4o                  | Input: $2.50, Output: $10.00 | Maximum accuracy            |
| Anthropic | claude-3-5-haiku        | Input: $0.80, Output: $4.00  | Fast, cost-effective        |
| Anthropic | claude-3-5-sonnet       | Input: $3.00, Output: $15.00 | Previous generation balance |
| Anthropic | claude-sonnet-4-5 (NEW) | Input: $3.00, Output: $15.00 | Latest, best reasoning      |

## Performance Notes

- **OpenAI GPT-4o-mini**: Best cost-performance ratio, recommended for most use cases
- **Anthropic Claude Sonnet 4.5**: Latest model with excellent reasoning, may be better for complex brand name disambiguation
- **Anthropic Claude 3.5 Sonnet**: Previous generation, still excellent for most tasks
- **Anthropic Claude Haiku**: Fastest responses, good for high-volume processing

## Architecture

```
YOLO Detection → GPT-4 Vision OCR → Hybrid Matching
                                      ├─ Exact Match (100%)
                                      ├─ Fuzzy+Visual (85-100%)
                                      └─ LLM Verification (70-84%)
                                          ├─ OpenAI GPT-4
                                          └─ Anthropic Claude
```

## Files

- `llm_verifier.py` - OpenAI GPT-4 verifier
- `llm_verifier_anthropic.py` - Anthropic Claude verifier
- `llm_verifier_unified.py` - Unified interface for both providers
- `test.py` - Main application with multi-provider support
- `requirements.txt` - Dependencies for all providers

## Troubleshooting

### Missing API Key Error

If you see `ANTHROPIC_API_KEY not found`, add it to your `.env` file or stick with OpenAI.

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Rate Limiting

Both providers have rate limits. The system includes retry logic with exponential backoff.
