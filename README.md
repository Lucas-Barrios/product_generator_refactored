# AI Product Listing Generator

Automated product listing generator using GPT-4 Vision API.
Given a product image and basic metadata, generates professional 
e-commerce listings in JSON format.

## Stack
- Python 3
- OpenAI GPT-4o / Vision API
- HuggingFace `datasets`
- Pillow
- Pandas
- Pydantic
- python-dotenv
- Python logging

## Setup
1. Clone the repo
2. Create venv and activate it
3. Run `pip install -r requirements.txt`
4. Add your `OPENAI_API_KEY` to a `.env` file
5. Run `python product_generator.py`

## Refactored Structure

- `product_generator.py` - main refactored product listing generator.
- `openai_wrapper.py` - reusable OpenAI API wrapper with retry logic and exponential backoff.
- `product_generator_original.py` - original version kept for comparison.
- `before_after_comparison.md` - documents how the original code changed after refactoring.
- `lab_summary.md` - short summary of the refactoring work and takeaways.
- `outputs/` - proof artifacts showing successful processing and error handling examples.

## Improvements Added

- Modular helper functions for loading, validation, prompt creation, API calls, response parsing, output formatting, saving, and batch processing.
- Pydantic validation for product input data and generated listing responses.
- Contextual error messages that show the function, error type, location, message, and suggestion.
- Reusable API wrapper with standardized success/error responses and retry logic.
- Logging audit trail written to `product_generator.log`.

## Proof Outputs

The `outputs/` folder includes examples showing:

- successful processing with valid data
- error messages for invalid product data
- error messages for missing files
- error messages for API failures
