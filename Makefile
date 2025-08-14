# Makefile for PaDiff
#
# 	GitHb: https://github.com/PaddlePaddle/PaDiff
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test
check_dirs := padiff tests scripts
# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	pre-commit run black

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

.PHONY: lint lint-all
lint:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi	

lint-all:
	pre-commit run --all-files

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	@echo "Running unit tests with coverage..."
	PYTHONPATH=. coverage run --source=. tests/padiff_unittests.py
	@echo ""
	@echo "Coverage Report:"
	coverage report -m
	@echo ""
	@echo "Generating XML report for CI..."
	coverage xml
	@echo "Coverage report generated: coverage.xml"

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	pip install --upgrade pip
	pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
	pre-commit install
