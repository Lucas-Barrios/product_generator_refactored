# Error Messages For API Errors

The following output proves that API failures show where the error occurred, which retry attempt failed, the model settings used, the final failure location, and a helpful suggestion.

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

This confirms API errors are not hidden: each failed retry is reported, and the final failure identifies that the listing API call failed after all retry attempts were exhausted.

