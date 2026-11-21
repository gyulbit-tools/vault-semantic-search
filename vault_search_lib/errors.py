"""One error type with fixed codes.

Messages are fixed codes (and, for configuration errors, the name of the bad key or a fixed reason),
never note paths, queries or note contents, so a failed run does not leak them
into logs or scheduler output.
"""


class SearchError(ValueError):
    def __init__(self, code, detail=""):
        super().__init__(code)
        self.code = code
        self.detail = detail
