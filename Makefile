# Makefile

VENV_DIR=env

.PHONY: setup activate test clean

setup:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	$(VENV_DIR)/bin/pip install --only-binary=:all: matplotlib
	$(VENV_DIR)/bin/pip install -r requirements.txt
	$(VENV_DIR)/bin/pip install ipython


activate:
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"

test:
	$(VENV_DIR)/bin/ipython scripts/train_mnist_jacobian_attack.py -- --max_epochs 2 --num_samples 2

clean:
	rm -rf $(VENV_DIR)
