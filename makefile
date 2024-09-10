# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt

# Default target
.PHONY: all
all: venv install run

# Create virtual environment
.PHONY: venv
venv:
	virtualenv $(VENV_DIR)

# Install dependencies
.PHONY: install
install: venv
	$(PIP) install -r $(REQUIREMENTS)

# Run image resizing script
.PHONY: resize
resize: install
	$(PYTHON) resize_images.py

# Run palette creation script
.PHONY: palette
palette: install
	$(PYTHON) create_palettes.py

# Run feature extraction script
.PHONY: features
features: install
	$(PYTHON) get_features_and_labels.py

# Train model
.PHONY: train
train: install
	$(PYTHON) train.py

# Test model
.PHONY: test
test: install
	$(PYTHON) test.py

# Run all steps in order
.PHONY: run
run: resize palette features train test

# Clean up the virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
