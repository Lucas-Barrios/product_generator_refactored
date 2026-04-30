from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import base64
from io import BytesIO
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


DATASET_NAME = "ashraq/fashion-product-images-small"
DATASET_SPLIT = "train[:10]"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_PRICE = 49.99
DEFAULT_MAX_TOKENS = 1000
DEFAULT_NUM_PRODUCTS = 3
REQUEST_DELAY_SECONDS = 2
OUTPUT_PATH = Path("outputs/product_listings.json")


def print_error(
    function_name: str,
    error: Exception,
    location: str,
    suggestion: str,
    message: str | None = None,
) -> None:
    """Print a consistent, actionable error message."""
    error_type = type(error).__name__
    error_message = message or str(error)
    print(
        f"ERROR in {function_name}(): {error_type}\n"
        f"  Location: {location}\n"
        f"  Message: {error_message}\n"
        f"  Suggestion: {suggestion}"
    )


class ProductData(BaseModel):
    """Validated product fields required to generate a listing."""

    id: int
    product_name: str = Field(alias="productDisplayName")
    category: str = Field(alias="masterCategory")
    base_colour: str = Field(alias="baseColour")
    season: str
    usage: str
    price: float = DEFAULT_PRICE
    image: Any

    @property
    def additional_info(self) -> str:
        return f"{self.base_colour}, {self.season}, {self.usage}"


class ProductListing(BaseModel):
    """Validated structure expected from the OpenAI response."""

    title: str
    description: str
    features: list[str]
    keywords: str


def load_json_file(file_path: str | Path) -> dict:
    """Load and parse a JSON file with error handling."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError as error:
        print_error(
            "load_json_file",
            error,
            f"File '{file_path}' not found",
            "Check that the file path is correct and the file exists.",
        )
        raise
    except PermissionError as error:
        print_error(
            "load_json_file",
            error,
            f"File '{file_path}' cannot be read",
            "Check file permissions and make sure your user can read this file.",
        )
        raise
    except json.JSONDecodeError as error:
        print_error(
            "load_json_file",
            error,
            f"File '{file_path}', line {error.lineno}, column {error.colno}",
            "Check JSON syntax at the indicated location.",
            error.msg,
        )
        raise
    except OSError as error:
        print_error(
            "load_json_file",
            error,
            f"File '{file_path}'",
            "Check that the path is accessible and is a valid file path.",
        )
        raise


def save_json_file(data: Any, file_path: str | Path) -> None:
    """Save data to a JSON file with error handling."""
    try:
        output_path = Path(file_path)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as file:
            json.dump(data, file, indent=2)
    except PermissionError as error:
        print_error(
            "save_json_file",
            error,
            f"File '{file_path}' cannot be written",
            "Check output directory permissions or choose a writable path.",
        )
        raise
    except TypeError as error:
        print_error(
            "save_json_file",
            error,
            f"File '{file_path}', data type '{type(data).__name__}'",
            "Make sure the data only contains JSON-serializable values.",
        )
        raise
    except OSError as error:
        print_error(
            "save_json_file",
            error,
            f"File '{file_path}'",
            "Check that the output path is valid and the parent directory is writable.",
        )
        raise


def product_row_to_dict(product_row: Any) -> dict:
    """Convert a pandas row or dictionary-like product into a plain dict."""
    try:
        if hasattr(product_row, "to_dict"):
            return product_row.to_dict()
        return dict(product_row)
    except (TypeError, ValueError) as error:
        print_error(
            "product_row_to_dict",
            error,
            f"Input type '{type(product_row).__name__}'",
            "Pass a pandas row, dictionary, or dictionary-like product object.",
        )
        raise


def apply_price_override(product_dict: dict, price: float | None = None) -> dict:
    """Return product data with an optional price override."""
    product_data = product_dict.copy()
    if price is not None:
        product_data["price"] = price
    return product_data


def validate_product_data(product_dict: dict) -> bool:
    """Validate product data using Pydantic."""
    try:
        ProductData.model_validate(product_dict)
        return True
    except ValidationError as error:
        print_error(
            "validate_product_data",
            error,
            "ProductData schema validation",
            "Check required product fields: id, productDisplayName, masterCategory, baseColour, season, usage, image, and price.",
        )
        return False


def build_product_data(product_row: Any, price: float | None = None) -> ProductData:
    """Create validated product data from a dataset row."""
    try:
        if isinstance(product_row, ProductData):
            if price is None:
                return product_row
            product_row = product_row.model_dump(by_alias=True)

        product_dict = product_row_to_dict(product_row)
        product_dict = apply_price_override(product_dict, price)
        return ProductData.model_validate(product_dict)
    except ValidationError as error:
        print_error(
            "build_product_data",
            error,
            "Product row validation",
            "Check the product row contains valid dataset fields and compatible field types.",
        )
        raise


def encode_image(pil_image: Any) -> str:
    """Convert PIL image to base64 string for API transmission."""
    try:
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    except AttributeError as error:
        print_error(
            "encode_image",
            error,
            f"Input type '{type(pil_image).__name__}'",
            "Pass a valid PIL image object with a save() method.",
        )
        raise
    except OSError as error:
        print_error(
            "encode_image",
            error,
            "PIL image encoding as JPEG",
            "Check that the image is valid and can be saved as JPEG.",
        )
        raise


def create_product_prompt(product: ProductData | dict) -> str:
    """Generate OpenAI prompt for a product."""
    try:
        if isinstance(product, dict):
            product = ProductData.model_validate(product)
    except ValidationError as error:
        print_error(
            "create_product_prompt",
            error,
            "Product prompt input validation",
            "Pass a valid ProductData object or dictionary with all required product fields.",
        )
        raise

    return f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product.product_name}
- Price: ${product.price:.2f}
- Category: {product.category}
- Additional Info: {product.additional_info}

Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 keywords)

Format your response as JSON with this exact structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2"],
    "keywords": "keyword1, keyword2"
}}

Be specific about what you see in the image. Mention colors, materials,
design elements and distinctive features. Avoid generic descriptions."""


def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """Create a structured prompt for generating product listings."""
    product_dict = {
        "id": 0,
        "productDisplayName": product_name,
        "masterCategory": category,
        "baseColour": "",
        "season": "",
        "usage": additional_info or "",
        "price": price,
        "image": None,
    }
    return create_product_prompt(product_dict)


def extract_json_from_text(raw_text: str) -> str:
    """Extract JSON content from a plain or markdown-fenced response."""
    try:
        clean_response = raw_text.strip()
    except AttributeError as error:
        print_error(
            "extract_json_from_text",
            error,
            f"Response text type '{type(raw_text).__name__}'",
            "Pass response content as a string before parsing JSON.",
        )
        raise

    if "```json" in clean_response:
        return clean_response.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in clean_response:
        return clean_response.split("```", 1)[1].split("```", 1)[0].strip()

    return clean_response


def get_response_text(response: Any) -> str:
    """Read message content from an OpenAI response object or test dictionary."""
    try:
        if isinstance(response, dict):
            return response["choices"][0]["message"]["content"]
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        print_error(
            "get_response_text",
            error,
            "OpenAI response choices[0].message.content",
            "Check that the API response includes choices with message content.",
        )
        raise


def parse_api_response(response: Any) -> dict:
    """Parse and validate an OpenAI API response."""
    try:
        raw_response = get_response_text(response)
        json_text = extract_json_from_text(raw_response)
        listing = json.loads(json_text)
        return validate_listing_data(listing).model_dump()
    except json.JSONDecodeError as error:
        print_error(
            "parse_api_response",
            error,
            f"API response JSON, line {error.lineno}, column {error.colno}",
            "Check that the model returned valid JSON and did not include extra prose.",
            error.msg,
        )
        raise
    except ValidationError as error:
        print_error(
            "parse_api_response",
            error,
            "ProductListing schema validation",
            "Check that the response contains title, description, features, and keywords.",
        )
        raise


def validate_listing_data(listing_dict: dict) -> ProductListing:
    """Validate generated product listing data."""
    try:
        return ProductListing.model_validate(listing_dict)
    except ValidationError as error:
        print_error(
            "validate_listing_data",
            error,
            "ProductListing schema validation",
            "Check that listing data contains title, description, features, and keywords with valid types.",
        )
        raise


def format_output(product: ProductData | dict, result: dict) -> dict:
    """Format final output for one processed product."""
    try:
        if isinstance(product, dict):
            product = ProductData.model_validate(product)

        output = {
            "id": product.id,
            "product_name": product.product_name,
            "category": product.category,
            "status": result["status"],
        }

        if result["status"] == "success":
            output["listing"] = result["listing"]
        else:
            output["error"] = result["error"]

        return output
    except ValidationError as error:
        print_error(
            "format_output",
            error,
            "Product output formatting",
            "Pass a valid ProductData object or product dictionary.",
        )
        raise
    except KeyError as error:
        print_error(
            "format_output",
            error,
            f"Missing result key {error}",
            "Result must include status plus listing for success or error for failures.",
        )
        raise


def create_messages(encoded_image: str, prompt: str) -> list[dict]:
    """Build OpenAI chat messages for an image and prompt."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def build_product_messages(product: ProductData) -> list[dict]:
    """Create API messages for one product."""
    encoded_image = encode_image(product.image)
    prompt = create_product_prompt(product)
    return create_messages(encoded_image, prompt)


def call_listing_api(
    client: OpenAI,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    """Call the OpenAI API for a product listing."""
    try:
        return client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
    except Exception as error:
        print_error(
            "call_listing_api",
            error,
            f"OpenAI chat.completions.create model='{model}', max_tokens={max_tokens}",
            "Check your API key, network connection, model name, quota, and rate limits.",
        )
        raise


def create_success_result(listing: dict) -> dict:
    """Create a successful processing result."""
    return {"status": "success", "listing": listing}


def create_error_result(error: Exception) -> dict:
    """Create a failed processing result."""
    return {"status": "error", "error": str(error)}


def generate_listing_for_product(
    product: ProductData,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Generate a listing for one validated product."""
    messages = build_product_messages(product)
    response = call_listing_api(client, messages, model, max_tokens)
    listing = parse_api_response(response)
    return create_success_result(listing)


def safely_generate_listing_for_product(
    product: ProductData,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Generate a listing and convert failures into result data."""
    try:
        return generate_listing_for_product(product, client, model, max_tokens)
    except Exception as error:
        return create_error_result(error)


def generate_product_listing(
    product_row: Any,
    client: OpenAI,
    price: float | None = DEFAULT_PRICE,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Build product data and generate a listing."""
    try:
        product = build_product_data(product_row, price)
        return generate_listing_for_product(product, client, model, max_tokens)

    except Exception as error:
        return create_error_result(error)


def load_product_dataset_split(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> Any:
    """Load product data from HuggingFace."""
    try:
        return load_dataset(dataset_name, split=split)
    except Exception as error:
        print_error(
            "load_product_dataset_split",
            error,
            f"Dataset '{dataset_name}', split '{split}'",
            "Check the dataset name, split string, internet connection, and HuggingFace availability.",
        )
        raise


def dataset_to_dataframe(dataset: Any) -> pd.DataFrame:
    """Convert a dataset into a DataFrame."""
    try:
        return pd.DataFrame(dataset)
    except Exception as error:
        print_error(
            "dataset_to_dataframe",
            error,
            f"Dataset type '{type(dataset).__name__}'",
            "Check that the loaded dataset is tabular or convertible to a pandas DataFrame.",
        )
        raise


def load_product_dataset(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> pd.DataFrame:
    """Load product data and return it as a DataFrame."""
    dataset = load_product_dataset_split(dataset_name, split)
    return dataset_to_dataframe(dataset)


def summarize_results(results: list[dict]) -> dict:
    """Summarize successful and failed product results."""
    return {
        "processed": len(results),
        "successful": sum(1 for result in results if result["status"] == "success"),
        "failed": sum(1 for result in results if result["status"] == "error"),
    }


def get_product_rows(df: pd.DataFrame, num_products: int) -> list[Any]:
    """Return the product rows selected for processing."""
    try:
        total = min(num_products, len(df))
        return [df.iloc[i] for i in range(total)]
    except (AttributeError, TypeError, ValueError) as error:
        print_error(
            "get_product_rows",
            error,
            f"DataFrame type '{type(df).__name__}', num_products={num_products}",
            "Pass a pandas DataFrame and a non-negative integer product count.",
        )
        raise


def process_product(product: ProductData, client: OpenAI) -> dict:
    """Process one validated product."""
    result = safely_generate_listing_for_product(product, client)
    return format_output(product, result)


def process_product_row(product_row: Any, client: OpenAI) -> dict:
    """Validate and process one product row."""
    try:
        product = build_product_data(product_row)
        return process_product(product, client)
    except Exception as error:
        print_error(
            "process_product_row",
            error,
            "Single product row processing",
            "Check the product row fields, image value, API client, and API response format.",
        )
        raise


def wait_before_next_request(current_index: int, total: int, request_delay: int) -> None:
    """Pause between API requests."""
    if current_index < total - 1:
        time.sleep(request_delay)


def print_processing_header(total: int) -> None:
    """Print the batch processing header."""
    print(f"\nProcessing {total} products...")
    print("=" * 50)


def print_product_progress(index: int, total: int, product_name: str) -> None:
    """Print progress for one product."""
    print(f"\n[{index + 1}/{total}] Processing: {product_name}")


def print_result_status(result: dict) -> None:
    """Print the status for one generated listing."""
    if result["status"] == "success":
        print("✓ Listing generated successfully")
    else:
        print(f"✗ Failed: {result['error']}")


def print_wait_message(current_index: int, total: int, request_delay: int) -> None:
    """Print the request delay message."""
    if current_index < total - 1:
        print(f"  Waiting {request_delay} seconds before next request...")


def print_summary(summary: dict, output_path: Path) -> None:
    """Print the batch processing summary."""
    print("\n" + "=" * 50)
    print(f"✓ Processed {summary['processed']} products")
    print(f"✓ Successful: {summary['successful']}")
    print(f"✗ Failed: {summary['failed']}")
    print(f"✓ Results saved to {output_path}")


def process_multiple_products(
    df: pd.DataFrame,
    client: OpenAI,
    num_products: int = DEFAULT_NUM_PRODUCTS,
    request_delay: int = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Process multiple products."""
    results = []
    product_rows = get_product_rows(df, num_products)

    for i, product_row in enumerate(product_rows):
        product = build_product_data(product_row)
        results.append(process_product(product, client))
        wait_before_next_request(i, len(product_rows), request_delay)

    return results


def save_results(results: list[dict], output_path: str | Path = OUTPUT_PATH) -> None:
    """Save product listing results."""
    save_json_file(results, output_path)


def run_product_batch(
    df: pd.DataFrame,
    client: OpenAI,
    num_products: int = DEFAULT_NUM_PRODUCTS,
    output_path: Path = OUTPUT_PATH,
    request_delay: int = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Run product processing, reporting, and saving."""
    results = []
    product_rows = get_product_rows(df, num_products)
    total = len(product_rows)

    print_processing_header(total)

    for i, product_row in enumerate(product_rows):
        product = build_product_data(product_row)
        print_product_progress(i, total, product.product_name)

        formatted_result = process_product(product, client)
        results.append(formatted_result)

        print_result_status(formatted_result)
        print_wait_message(i, total, request_delay)
        wait_before_next_request(i, total, request_delay)

    save_results(results, output_path)

    summary = summarize_results(results)
    print_summary(summary, output_path)

    return results


def create_openai_client() -> OpenAI:
    """Initialize the OpenAI client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error = ValueError("OPENAI_API_KEY is not set")
        print_error(
            "create_openai_client",
            error,
            "Environment variable OPENAI_API_KEY",
            "Add OPENAI_API_KEY to your .env file or shell environment.",
        )
        raise error
    try:
        return OpenAI(api_key=api_key)
    except Exception as error:
        print_error(
            "create_openai_client",
            error,
            "OpenAI client initialization",
            "Check that the OpenAI package is installed and the API key value is valid.",
        )
        raise


def main() -> None:
    """Run batch product listing generation."""
    load_dotenv()

    client = create_openai_client()
    print("✓ API client initialized successfully")

    print("\nLoading product dataset...")
    products_df = load_product_dataset()
    print(f"✓ Loaded {len(products_df)} products")

    run_product_batch(products_df, client, num_products=DEFAULT_NUM_PRODUCTS)


if __name__ == "__main__":
    main()
