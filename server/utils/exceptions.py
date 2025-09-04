class BenchmarkError(Exception):
    """Base exception for benchmark operations"""
    pass

class DatabaseError(Exception):
    """Database operation exception"""
    pass

class ValidationError(Exception):
    """Input validation exception"""
    pass

class BenchmarkExecutionError(BenchmarkError):
    """Benchmark execution failure exception"""
    pass

class URLValidationError(ValidationError):
    """URL validation failure exception"""
    pass