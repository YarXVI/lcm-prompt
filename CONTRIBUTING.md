# Contributing to LCM Protocol

Thank you for your interest in contributing to LCM Protocol! This document provides guidelines for contributing to the project.

## Development Setup

```bash
git clone https://github.com/iris-team/lcm-protocol.git
cd lcm-protocol
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lcm_v2

# Run specific test file
pytest tests/test_lcm_v2.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to public functions and classes
- Keep functions focused and concise

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request

## Reporting Issues

When reporting issues, please include:
- Python version
- LCM version
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Code of Conduct

Be respectful and constructive in all interactions.
