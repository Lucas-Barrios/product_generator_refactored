# Error Message For Missing Files

The following output proves that a missing file clearly shows where the error occurred, the missing path, the error type, and a helpful suggestion.

```text
ERROR in load_json_file(): FileNotFoundError
  Location: File '/var/folders/.../missing.json' not found
  Message: [Errno 2] No such file or directory: '/var/folders/.../missing.json'
  Suggestion: Check that the file path is correct and the file exists.
```

This confirms the missing-file path is reported by `load_json_file()` instead of failing silently.

