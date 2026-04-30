# Logging Examples

This file contains representative examples from `product_generator.log` showing the audit trail created by the refactored product generator.

## Successful Processing

```text
2026-04-30 13:54:29,866 - product_generator - INFO - Starting product batch run for up to 1 product(s)
2026-04-30 13:54:29,868 - product_generator - INFO - Built ProductData for id=101, name=Test Black Shirt
2026-04-30 13:54:29,879 - product_generator - INFO - OpenAI chat completion succeeded on attempt 1
2026-04-30 13:54:29,880 - product_generator - INFO - Generated listing successfully for product id=101
2026-04-30 13:54:29,881 - product_generator - INFO - Successfully saved JSON file to /var/folders/.../valid_product_listings.json
2026-04-30 13:54:29,881 - product_generator - INFO - Processing summary: {'processed': 1, 'successful': 1, 'failed': 0}
2026-04-30 13:54:29,882 - product_generator - INFO - Completed product batch run
```

## Missing File And Invalid JSON

```text
2026-04-30 12:50:06,975 - product_generator - INFO - Loading JSON file from /var/folders/.../missing.json
2026-04-30 12:50:06,976 - product_generator - ERROR - load_json_file failed with FileNotFoundError at File '/var/folders/.../missing.json' not found: [Errno 2] No such file or directory | Suggestion: Check that the file path is correct and the file exists.
2026-04-30 12:50:06,976 - product_generator - INFO - Loading JSON file from /var/folders/.../invalid.json
2026-04-30 12:50:06,976 - product_generator - ERROR - load_json_file failed with JSONDecodeError at File '/var/folders/.../invalid.json', line 2, column 3: Expecting property name enclosed in double quotes | Suggestion: Check JSON syntax at the indicated location.
```

## Product Validation Error

```text
2026-04-30 12:50:06,977 - product_generator - ERROR - validate_product_data failed with ValidationError at ProductData schema validation: 6 validation errors for ProductData
id
  Field required
masterCategory
  Field required
baseColour
  Field required
season
  Field required
usage
  Field required
image
  Field required
```

## API Retry And Failure

```text
2026-04-30 12:50:06,978 - product_generator - INFO - Sending OpenAI chat completion request with model=gpt-4o, max_tokens=1000, attempt=1/2
2026-04-30 12:50:06,979 - product_generator - ERROR - OpenAIWrapper.create_chat_completion failed with RuntimeError at Attempt 1/2, model='gpt-4o', max_tokens=1000: simulated API rate limit | Suggestion: Check your API key, network connection, model name, quota, and rate limits.
2026-04-30 12:50:06,979 - product_generator - INFO - Retrying OpenAI request in 0 seconds
2026-04-30 12:50:06,980 - product_generator - ERROR - OpenAI chat completion failed after 2 attempts
2026-04-30 12:50:06,980 - product_generator - ERROR - call_listing_api failed with RuntimeError at OpenAI API failed after 2 attempt(s): simulated API rate limit | Suggestion: Check your API key, network connection, model name, quota, and rate limits.
```

## Why These Logs Are Useful

These logs make it easier to debug the product generator because they show when processing starts and finishes, which product is being processed, when API calls succeed or retry, where file and JSON errors occur, and which validation fields are missing. The result is a readable audit trail that explains both successful runs and failures.
