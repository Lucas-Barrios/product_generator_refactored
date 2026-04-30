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

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✓ API client initialized successfully")

# Load dataset from HuggingFace
print("\nLoading product dataset...")
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train[:10]"
)
products_df = pd.DataFrame(dataset)
print(f"✓ Loaded {len(products_df)} products")

def encode_image(pil_image):
    """Convert PIL image to base64 string for API transmission."""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """Create a structured prompt for generating product listings."""
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Info: {additional_info}' if additional_info else ''}

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
    return prompt

def generate_product_listing(product_row, price=49.99):
    """Send image and metadata to GPT-4 Vision and get product listing."""
    try:
        encoded_image = encode_image(product_row["image"])
        prompt = create_product_listing_prompt(
            product_name=product_row["productDisplayName"],
            price=price,
            category=product_row["masterCategory"],
            additional_info=f"{product_row['baseColour']}, {product_row['season']}, {product_row['usage']}"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        raw_response = response.choices[0].message.content
        clean_response = raw_response.strip()

        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_response:
            clean_response = clean_response.split("```")[1].split("```")[0].strip()

        listing = json.loads(clean_response)
        return {"status": "success", "listing": listing}

    except Exception as e:
        return {"status": "error", "error": str(e)}

def process_multiple_products(df, num_products=3):
    """Generate listings for multiple products and save results."""
    results = []
    total = min(num_products, len(df))

    print(f"\nProcessing {total} products...")
    print("=" * 50)

    for i in range(total):
        product = df.iloc[i]
        product_name = product["productDisplayName"]

        print(f"\n[{i+1}/{total}] Processing: {product_name}")

        result = generate_product_listing(product)

        if result["status"] == "success":
            print(f"✓ Listing generated successfully")
            results.append({
                "id": int(product["id"]),
                "product_name": product_name,
                "category": product["masterCategory"],
                "status": "success",
                "listing": result["listing"]
            })
        else:
            print(f"✗ Failed: {result['error']}")
            results.append({
                "id": int(product["id"]),
                "product_name": product_name,
                "category": product["masterCategory"],
                "status": "error",
                "error": result["error"]
            })

        if i < total - 1:
            print("  Waiting 2 seconds before next request...")
            time.sleep(2)

    output_path = Path("outputs/product_listings.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"✓ Processed {len(results)} products")
    print(f"✓ Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"✗ Failed: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"✓ Results saved to {output_path}")

    return results

# Run batch processing
results = process_multiple_products(products_df, num_products=3)