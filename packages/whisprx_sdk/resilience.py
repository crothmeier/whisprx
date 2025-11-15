#!/usr/bin/env python3
"""Resilience utilities for production-grade error handling"""
import asyncio
import functools
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests rejected immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call()"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker CLOSED (recovered)")

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker OPEN ({self.failure_count} failures)")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery"""
        if self.last_failure_time is None:
            return True

        current_time = asyncio.get_event_loop().time()
        return (current_time - self.last_failure_time) >= self.recovery_timeout


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
):
    """Decorator for retrying functions with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Use exponential backoff (vs linear)

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def unreliable_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        # Calculate delay
                        if exponential:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                        else:
                            delay = min(base_delay * (attempt + 1), max_delay)

                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s "
                            f"due to: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Failed after {max_retries} retries: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


class HealthCheck:
    """Health check manager for monitoring service health"""

    def __init__(self):
        self.checks = {}
        self.status = "healthy"

    def register(self, name: str, check_func: Callable):
        """Register a health check function

        Args:
            name: Check name
            check_func: Async function that returns True if healthy
        """
        self.checks[name] = check_func

    async def run_checks(self) -> dict:
        """Run all health checks

        Returns:
            Dictionary with check results
        """
        results = {}
        all_healthy = True

        for name, check_func in self.checks.items():
            try:
                is_healthy = await check_func()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "healthy": is_healthy,
                }

                if not is_healthy:
                    all_healthy = False

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e),
                }
                all_healthy = False

        self.status = "healthy" if all_healthy else "unhealthy"

        return {
            "status": self.status,
            "checks": results,
        }


class GracefulShutdown:
    """Graceful shutdown manager for cleanup on termination"""

    def __init__(self):
        self.shutdown_handlers = []
        self.is_shutting_down = False

    def register(self, handler: Callable):
        """Register a shutdown handler

        Args:
            handler: Async function to call on shutdown
        """
        self.shutdown_handlers.append(handler)

    async def shutdown(self):
        """Execute all shutdown handlers"""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        logger.info("Starting graceful shutdown...")

        for handler in self.shutdown_handlers:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")

        logger.info("Graceful shutdown complete")


# Global instances
_health_check = HealthCheck()
_shutdown_manager = GracefulShutdown()


def get_health_check() -> HealthCheck:
    """Get global health check instance"""
    return _health_check


def get_shutdown_manager() -> GracefulShutdown:
    """Get global shutdown manager"""
    return _shutdown_manager


# Example usage
if __name__ == "__main__":
    import asyncio

    # Circuit breaker example
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)

    async def unreliable_service():
        import random
        if random.random() < 0.5:
            raise Exception("Service failed")
        return "Success"

    # Retry example
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def flaky_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"

    async def main():
        # Test retry
        try:
            result = await flaky_function()
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed: {e}")

        # Test health check
        health = get_health_check()

        async def check_database():
            return True  # Simulate database check

        health.register("database", check_database)

        results = await health.run_checks()
        print(f"Health: {results}")

    asyncio.run(main())
