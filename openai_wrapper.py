import time
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1


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


def get_response_text(response: Any) -> str:
    """Read message content from an OpenAI response object or test dictionary."""
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content


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
                response = self.client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
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
                    time.sleep(self.get_retry_delay(attempt))

        return self.create_error_response(last_error, self.max_retries)

    def generate_description(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        """Generate text from a prompt with retry logic."""
        messages = [{"role": "user", "content": prompt}]
        api_result = self.create_chat_completion(messages, model, max_tokens)

        if api_result["status"] == "error":
            return api_result

        try:
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
