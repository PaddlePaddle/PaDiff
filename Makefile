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

.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi	

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	PYTHONPATH=. PADIFF_API_CHECK=ON python tests/test_api_to_Layer.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_check_weight_grad.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_compare_mode.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_device.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_diff_phase.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_inplace.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_layer_map.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_loss_fn.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_multi_input.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_optimizer_multi_steps.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_resnet50.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_simplenet1.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_simplenet2.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_simplenet3.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_simplenet4.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_simplenet5.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_single_step.py
	PYTHONPATH=. PADIFF_API_CHECK=OFF python tests/test_tree_view.py

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
	pre-commit install
