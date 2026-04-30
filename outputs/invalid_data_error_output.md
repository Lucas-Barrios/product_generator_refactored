# Error Messages For Invalid Data

The following output proves that invalid inputs show where errors occur, the error type, the context, the message, and a helpful suggestion.

## Missing File

```text
ERROR in load_json_file(): FileNotFoundError
  Location: File '/var/folders/.../missing.json' not found
  Message: [Errno 2] No such file or directory: '/var/folders/.../missing.json'
  Suggestion: Check that the file path is correct and the file exists.
```

## Invalid JSON

```text
ERROR in load_json_file(): JSONDecodeError
  Location: File '/var/folders/.../invalid.json', line 2, column 3
  Message: Expecting property name enclosed in double quotes
  Suggestion: Check JSON syntax at the indicated location.
```

## Invalid Product Data

```text
ERROR in validate_product_data(): ValidationError
  Location: ProductData schema validation
  Message: 6 validation errors for ProductData
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
  Suggestion: Check required product fields: id, productDisplayName, masterCategory, baseColour, season, usage, image, and price.
```

## API Error

```text
ERROR in OpenAIWrapper.create_chat_completion(): RuntimeError
  Location: Attempt 1/2, model='gpt-4o', max_tokens=1000
  Message: simulated API rate limit
  Suggestion: Check your API key, network connection, model name, quota, and rate limits.

ERROR in OpenAIWrapper.create_chat_completion(): RuntimeError
  Location: Attempt 2/2, model='gpt-4o', max_tokens=1000
  Message: simulated API rate limit
  Suggestion: Check your API key, network connection, model name, quota, and rate limits.

ERROR in call_listing_api(): RuntimeError
  Location: OpenAI API failed after 2 attempt(s)
  Message: simulated API rate limit
  Suggestion: Check your API key, network connection, model name, quota, and rate limits.
```

