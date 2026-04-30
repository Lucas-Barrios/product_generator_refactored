from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import base64
from io import BytesIO
import json
import logging
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
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1
OUTPUT_PATH = Path("outputs/product_listings.json")
LOG_PATH = Path("product_generator.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
    logger.error(
        "%s failed with %s at %s: %s | Suggestion: %s",
        function_name,
        error_type,
        location,
        error_message,
        suggestion,
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


class OpenAIWrapper:
    """Wrapper for OpenAI API calls with error handling and retry logic."""

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: int | float = DEFAULT_BACKOFF_SECONDS,
        client: Any | None = None,
    ):
        if client is None and not api_key:
            error = ValueError("OPENAI_API_KEY is not set")
            print_error(
                "OpenAIWrapper.__init__",
                error,
                "OpenAI wrapper initialization",
                "Pass an API key or provide a preconfigured OpenAI-compatible client.",
            )
            raise error

        self.max_retries = max(1, max_retries)
        self.backoff_seconds = backoff_seconds
        self.client = client or OpenAI(api_key=api_key)
        logger.info(
            "OpenAIWrapper initialized with max_retries=%s and backoff_seconds=%s",
            self.max_retries,
            self.backoff_seconds,
        )

    def get_retry_delay(self, attempt: int) -> int | float:
        """Calculate exponential backoff delay for an attempt."""
        return self.backoff_seconds * (2 ** (attempt - 1))

    def create_success_response(self, response: Any, attempts: int) -> dict:
        """Create a standardized successful API response."""
        return {
            "status": "success",
            "response": response,
            "attempts": attempts,
        }

    def create_error_response(self, error: Exception, attempts: int) -> dict:
        """Create a standardized failed API response."""
        return {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "attempts": attempts,
        }

    def create_chat_completion(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        """Create a chat completion with retry logic."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Sending OpenAI chat completion request with model=%s, max_tokens=%s, attempt=%s/%s",
                    model,
                    max_tokens,
                    attempt,
                    self.max_retries,
                )
                logger.debug("OpenAI request messages: %s", messages)
                response = self.client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                logger.info("OpenAI chat completion succeeded on attempt %s", attempt)
                logger.debug("OpenAI raw response: %s", response)
                return self.create_success_response(response, attempt)
            except Exception as error:
                last_error = error
                print_error(
                    "OpenAIWrapper.create_chat_completion",
                    error,
                    f"Attempt {attempt}/{self.max_retries}, model='{model}', max_tokens={max_tokens}",
                    "Check your API key, network connection, model name, quota, and rate limits.",
                )

                if attempt < self.max_retries:
                    delay = self.get_retry_delay(attempt)
                    logger.info("Retrying OpenAI request in %s seconds", delay)
                    time.sleep(delay)

        logger.error("OpenAI chat completion failed after %s attempts", self.max_retries)
        return self.create_error_response(last_error, self.max_retries)

    def generate_description(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        """Generate text from a prompt with retry logic."""
        messages = [{"role": "user", "content": prompt}]
        logger.info("Generating description from prompt with length=%s", len(prompt))
        api_result = self.create_chat_completion(messages, model, max_tokens)

        if api_result["status"] == "error":
            logger.error("Description generation failed after %s attempts", api_result["attempts"])
            return api_result

        try:
            logger.info("Description generation succeeded after %s attempts", api_result["attempts"])
            return {
                "status": "success",
                "description": get_response_text(api_result["response"]),
                "attempts": api_result["attempts"],
            }
        except Exception as error:
            print_error(
                "OpenAIWrapper.generate_description",
                error,
                "OpenAI response text extraction",
                "Check that the API response includes choices[0].message.content.",
            )
            return self.create_error_response(error, api_result["attempts"])


def load_json_file(file_path: str | Path) -> dict:
    """Load and parse a JSON file with error handling."""
    try:
        logger.info("Loading JSON file from %s", file_path)
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.info("Successfully loaded JSON file from %s", file_path)
        logger.debug("Loaded JSON data: %s", data)
        return data
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
        logger.info("Saving JSON file to %s", output_path)
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as file:
            json.dump(data, file, indent=2)
        logger.info("Successfully saved JSON file to %s", output_path)
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
            product_dict = product_row.to_dict()
        else:
            product_dict = dict(product_row)
        logger.debug("Converted product row to dict with keys=%s", list(product_dict.keys()))
        return product_dict
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
        logger.info("Product data validation succeeded for id=%s", product_dict.get("id"))
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
                logger.debug("Received already validated ProductData for id=%s", product_row.id)
                return product_row
            product_row = product_row.model_dump(by_alias=True)

        product_dict = product_row_to_dict(product_row)
        product_dict = apply_price_override(product_dict, price)
        product = ProductData.model_validate(product_dict)
        logger.info("Built ProductData for id=%s, name=%s", product.id, product.product_name)
        return product
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
        logger.debug("Encoding image object of type %s", type(pil_image).__name__)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        logger.debug("Encoded image to base64 string with length=%s", len(encoded_image))
        return encoded_image
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

    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

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
    logger.debug("Created product prompt for id=%s with length=%s", product.id, len(prompt))
    return prompt


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
            content = response["choices"][0]["message"]["content"]
        else:
            content = response.choices[0].message.content
        logger.debug("Extracted response text with length=%s", len(content))
        return content
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
        logger.info("Parsing OpenAI API response")
        raw_response = get_response_text(response)
        json_text = extract_json_from_text(raw_response)
        listing = json.loads(json_text)
        validated_listing = validate_listing_data(listing).model_dump()
        logger.info("API response parsed and validated successfully")
        logger.debug("Parsed listing data: %s", validated_listing)
        return validated_listing
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
        listing = ProductListing.model_validate(listing_dict)
        logger.info("Product listing validation succeeded with title=%s", listing.title)
        return listing
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

        logger.debug("Formatted output for product id=%s with status=%s", product.id, result["status"])
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
    logger.info("Building API messages for product id=%s", product.id)
    encoded_image = encode_image(product.image)
    prompt = create_product_prompt(product)
    return create_messages(encoded_image, prompt)


def call_listing_api(
    api_client: Any,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Any:
    """Call the OpenAI API for a product listing through the wrapper."""
    logger.info("Calling listing API with model=%s, max_tokens=%s", model, max_tokens)
    wrapper = api_client if isinstance(api_client, OpenAIWrapper) else OpenAIWrapper(client=api_client)
    api_result = wrapper.create_chat_completion(messages, model, max_tokens)

    if api_result["status"] == "error":
        error = RuntimeError(api_result["error"])
        print_error(
            "call_listing_api",
            error,
            f"OpenAI API failed after {api_result['attempts']} attempt(s)",
            "Check your API key, network connection, model name, quota, and rate limits.",
            api_result["error"],
        )
        raise error

    logger.info("Listing API call completed after %s attempt(s)", api_result["attempts"])
    return api_result["response"]


def create_success_result(listing: dict) -> dict:
    """Create a successful processing result."""
    return {"status": "success", "listing": listing}


def create_error_result(error: Exception) -> dict:
    """Create a failed processing result."""
    return {"status": "error", "error": str(error)}


def generate_listing_for_product(
    product: ProductData,
    api_client: Any,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Generate a listing for one validated product."""
    logger.info("Generating listing for product id=%s, name=%s", product.id, product.product_name)
    messages = build_product_messages(product)
    response = call_listing_api(api_client, messages, model, max_tokens)
    listing = parse_api_response(response)
    logger.info("Generated listing successfully for product id=%s", product.id)
    return create_success_result(listing)


def safely_generate_listing_for_product(
    product: ProductData,
    api_client: Any,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Generate a listing and convert failures into result data."""
    try:
        return generate_listing_for_product(product, api_client, model, max_tokens)
    except Exception as error:
        logger.error("Listing generation failed for product id=%s: %s", product.id, error)
        return create_error_result(error)


def generate_product_listing(
    product_row: Any,
    api_client: Any,
    price: float | None = DEFAULT_PRICE,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Build product data and generate a listing."""
    try:
        product = build_product_data(product_row, price)
        return generate_listing_for_product(product, api_client, model, max_tokens)

    except Exception as error:
        return create_error_result(error)


def load_product_dataset_split(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> Any:
    """Load product data from HuggingFace."""
    try:
        logger.info("Loading product dataset '%s' with split '%s'", dataset_name, split)
        dataset = load_dataset(dataset_name, split=split)
        logger.info("Successfully loaded product dataset '%s'", dataset_name)
        return dataset
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
        logger.info("Converting dataset of type %s to DataFrame", type(dataset).__name__)
        df = pd.DataFrame(dataset)
        logger.info("Created DataFrame with %s rows and %s columns", len(df), len(df.columns))
        return df
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
    summary = {
        "processed": len(results),
        "successful": sum(1 for result in results if result["status"] == "success"),
        "failed": sum(1 for result in results if result["status"] == "error"),
    }
    logger.info("Processing summary: %s", summary)
    return summary


def get_product_rows(df: pd.DataFrame, num_products: int) -> list[Any]:
    """Return the product rows selected for processing."""
    try:
        total = min(num_products, len(df))
        logger.info("Selecting %s product row(s) from DataFrame with %s row(s)", total, len(df))
        return [df.iloc[i] for i in range(total)]
    except (AttributeError, TypeError, ValueError) as error:
        print_error(
            "get_product_rows",
            error,
            f"DataFrame type '{type(df).__name__}', num_products={num_products}",
            "Pass a pandas DataFrame and a non-negative integer product count.",
        )
        raise


def process_product(product: ProductData, api_client: Any) -> dict:
    """Process one validated product."""
    logger.info("Processing product id=%s", product.id)
    result = safely_generate_listing_for_product(product, api_client)
    formatted_result = format_output(product, result)
    logger.info("Finished processing product id=%s with status=%s", product.id, formatted_result["status"])
    return formatted_result


def process_product_row(product_row: Any, api_client: Any) -> dict:
    """Validate and process one product row."""
    try:
        product = build_product_data(product_row)
        return process_product(product, api_client)
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
        logger.info("Waiting %s seconds before next API request", request_delay)
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
    api_client: Any,
    num_products: int = DEFAULT_NUM_PRODUCTS,
    request_delay: int = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Process multiple products."""
    logger.info("Starting processing for up to %s product(s)", num_products)
    results = []
    product_rows = get_product_rows(df, num_products)

    for i, product_row in enumerate(product_rows):
        product = build_product_data(product_row)
        logger.info("Processing product %s/%s: %s", i + 1, len(product_rows), product.product_name)
        results.append(process_product(product, api_client))
        wait_before_next_request(i, len(product_rows), request_delay)

    logger.info("Finished processing %s product(s)", len(results))
    return results


def save_results(results: list[dict], output_path: str | Path = OUTPUT_PATH) -> None:
    """Save product listing results."""
    logger.info("Saving %s result(s) to %s", len(results), output_path)
    save_json_file(results, output_path)


def run_product_batch(
    df: pd.DataFrame,
    api_client: Any,
    num_products: int = DEFAULT_NUM_PRODUCTS,
    output_path: Path = OUTPUT_PATH,
    request_delay: int = REQUEST_DELAY_SECONDS,
) -> list[dict]:
    """Run product processing, reporting, and saving."""
    logger.info("Starting product batch run for up to %s product(s)", num_products)
    results = []
    product_rows = get_product_rows(df, num_products)
    total = len(product_rows)

    print_processing_header(total)

    for i, product_row in enumerate(product_rows):
        product = build_product_data(product_row)
        print_product_progress(i, total, product.product_name)

        formatted_result = process_product(product, api_client)
        results.append(formatted_result)

        print_result_status(formatted_result)
        print_wait_message(i, total, request_delay)
        wait_before_next_request(i, total, request_delay)

    save_results(results, output_path)

    summary = summarize_results(results)
    print_summary(summary, output_path)
    logger.info("Completed product batch run")

    return results


def create_openai_client() -> OpenAIWrapper:
    """Initialize the OpenAI API wrapper from environment variables."""
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
        logger.info("Creating OpenAI wrapper from environment variable OPENAI_API_KEY")
        wrapper = OpenAIWrapper(
            api_key=api_key,
            max_retries=DEFAULT_MAX_RETRIES,
            backoff_seconds=DEFAULT_BACKOFF_SECONDS,
        )
        logger.info("OpenAI wrapper created successfully")
        return wrapper
    except Exception as error:
        print_error(
            "create_openai_client",
            error,
            "OpenAI wrapper initialization",
            "Check that the OpenAI package is installed and the API key value is valid.",
        )
        raise


def main() -> None:
    """Run batch product listing generation."""
    load_dotenv()
    logger.info("Starting product generator")

    client = create_openai_client()
    print("✓ API client initialized successfully")

    print("\nLoading product dataset...")
    products_df = load_product_dataset()
    print(f"✓ Loaded {len(products_df)} products")

    run_product_batch(products_df, client, num_products=DEFAULT_NUM_PRODUCTS)
    logger.info("Product generator finished")


if __name__ == "__main__":
    main()
