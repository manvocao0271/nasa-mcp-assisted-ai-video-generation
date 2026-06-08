"""Custom exceptions wrapping failures from NASA APIs and the cache."""


class NasaMcpError(Exception):
    """Base exception for all nasa-mcp errors."""


class NasaApiError(NasaMcpError):
    """A NASA API returned an error response."""


class RateLimitError(NasaApiError):
    """The NASA API rate limit was exceeded."""


class NotFoundError(NasaApiError):
    """The requested resource does not exist."""


class CacheError(NasaMcpError):
    """The cache failed to read or write."""
    