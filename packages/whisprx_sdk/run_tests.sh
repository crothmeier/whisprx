#!/bin/bash
# Run pytest for async_pipeline module

# Install pytest and pytest-asyncio if not already installed
pip install pytest pytest-asyncio pytest-mock

# Run tests with coverage if pytest-cov is available
if python -c "import pytest_cov" 2>/dev/null; then
    pytest test_async_pipeline.py --cov=async_pipeline --cov-report=term-missing
else
    pytest test_async_pipeline.py
fi