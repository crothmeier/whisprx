import pytest
import asyncio
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.get_event_loop_policy()


@pytest.fixture
def event_loop(event_loop_policy):
    """Create an event loop for each test."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()