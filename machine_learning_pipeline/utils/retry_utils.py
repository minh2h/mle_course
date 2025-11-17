#!/usr/bin/env python3
"""
Retry and Error Recovery Utilities for ML Pipeline

This module provides robust retry mechanisms with exponential backoff,
circuit breaker patterns, and comprehensive error handling for the ML pipeline.
"""

import time
import random
import logging
import functools
from typing import Callable, Any, Optional, Tuple, Type, Union, List
from datetime import datetime, timedelta
from enum import Enum

class RetryStrategy(Enum):
    """Retry strategy enumeration."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"

class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class RetryableError(Exception):
    """Base class for retryable errors."""
    pass

class NonRetryableError(Exception):
    """Base class for non-retryable errors."""
    pass

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        self.max_attempts = max_attempts
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        self.non_retryable_exceptions = non_retryable_exceptions or (NonRetryableError,)

class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class RetryManager:
    """Advanced retry manager with multiple strategies and circuit breaker support."""
    
    def __init__(self, config: RetryConfig, circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        self.config = config
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config) if circuit_breaker_config else None
        self.logger = logging.getLogger(f"{__name__}.RetryManager")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic and optional circuit breaker."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    self.logger.error(f"Non-retryable error on attempt {attempt}: {e}")
                    raise e
                
                # Log retry attempt
                self.logger.warning(f"Attempt {attempt}/{self.config.max_attempts} failed: {e}")
                
                # Don't sleep after the last attempt
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
        
        # All attempts failed
        self.logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_exception
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable."""
        # Check non-retryable exceptions first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions
        return isinstance(exception, self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_factor ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.base_delay * (self.config.backoff_factor ** (attempt - 1))
            if self.config.jitter:
                # Add random jitter (±25%)
                jitter_range = base_delay * 0.25
                delay = base_delay + random.uniform(-jitter_range, jitter_range)
            else:
                delay = base_delay
        
        else:
            delay = self.config.base_delay
        
        # Cap the delay at max_delay
        return min(delay, self.config.max_delay)

def retry(
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    circuit_breaker: bool = False,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator for adding retry logic to functions."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_config = RetryConfig(
                max_attempts=max_attempts,
                strategy=strategy,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                non_retryable_exceptions=non_retryable_exceptions
            )
            
            circuit_breaker_config = None
            if circuit_breaker:
                circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout
                )
            
            retry_manager = RetryManager(retry_config, circuit_breaker_config)
            return retry_manager.execute(func, *args, **kwargs)
        
        return wrapper
    return decorator

# Predefined retry configurations for common scenarios
class CommonRetryConfigs:
    """Common retry configurations for different scenarios."""
    
    @staticmethod
    def database_operations() -> RetryConfig:
        """Retry config for database operations."""
        return RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=0.5,
            max_delay=30.0,
            backoff_factor=2.0,
            retryable_exceptions=(ConnectionError, TimeoutError),
            non_retryable_exceptions=(ValueError, TypeError)
        )
    
    @staticmethod
    def file_operations() -> RetryConfig:
        """Retry config for file I/O operations."""
        return RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            retryable_exceptions=(IOError, OSError),
            non_retryable_exceptions=(FileNotFoundError, PermissionError)
        )
    
    @staticmethod
    def network_requests() -> RetryConfig:
        """Retry config for network requests."""
        return RetryConfig(
            max_attempts=4,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=1.0,
            max_delay=60.0,
            backoff_factor=2.0,
            jitter=True
        )
    
    @staticmethod
    def model_training() -> RetryConfig:
        """Retry config for model training operations."""
        return RetryConfig(
            max_attempts=2,
            strategy=RetryStrategy.FIXED,
            base_delay=5.0,
            retryable_exceptions=(RuntimeError, MemoryError),
            non_retryable_exceptions=(ValueError, TypeError)
        )

# Convenience decorators for common scenarios
def retry_database_operation(**kwargs):
    """Decorator for database operations with sensible defaults."""
    config = CommonRetryConfigs.database_operations()
    return retry(
        max_attempts=config.max_attempts,
        strategy=config.strategy,
        base_delay=config.base_delay,
        max_delay=config.max_delay,
        backoff_factor=config.backoff_factor,
        retryable_exceptions=config.retryable_exceptions,
        non_retryable_exceptions=config.non_retryable_exceptions,
        **kwargs
    )

def retry_file_operation(**kwargs):
    """Decorator for file operations with sensible defaults."""
    config = CommonRetryConfigs.file_operations()
    return retry(
        max_attempts=config.max_attempts,
        strategy=config.strategy,
        base_delay=config.base_delay,
        max_delay=config.max_delay,
        backoff_factor=config.backoff_factor,
        retryable_exceptions=config.retryable_exceptions,
        non_retryable_exceptions=config.non_retryable_exceptions,
        **kwargs
    )

def retry_network_request(**kwargs):
    """Decorator for network requests with sensible defaults."""
    config = CommonRetryConfigs.network_requests()
    return retry(
        max_attempts=config.max_attempts,
        strategy=config.strategy,
        base_delay=config.base_delay,
        max_delay=config.max_delay,
        backoff_factor=config.backoff_factor,
        jitter=config.jitter,
        **kwargs
    )

# Example usage functions
def example_database_operation():
    """Example of using retry decorator for database operations."""
    
    @retry_database_operation()
    def connect_to_database():
        # Simulate database connection that might fail
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Database connection failed")
        return "Connected successfully"
    
    return connect_to_database()

def example_file_operation():
    """Example of using retry decorator for file operations."""
    
    @retry_file_operation()
    def read_file(filename: str):
        # Simulate file read that might fail
        import random
        if random.random() < 0.5:  # 50% chance of failure
            raise IOError("File read failed")
        return f"File {filename} read successfully"
    
    return read_file("example.txt")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Testing database operation retry:")
    try:
        result = example_database_operation()
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    print("\nTesting file operation retry:")
    try:
        result = example_file_operation()
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")