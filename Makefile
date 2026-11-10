.PHONY: test

# Run all tests under tests/
test:
	PYTHONPATH=src python -m pytest tests/ -q
