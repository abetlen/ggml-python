submodules = vendor/ggml

all: build

${submodules}:
	git submodule update --init --recursive

update-pip:
	python3 -m pip install --upgrade pip

build: ${submodules} update-pip ## Build ggml-python with cpu support
	python3 -m pip install --verbose --editable .

build.debug: ${submodules} update-pip ## Build ggml-python with cpu support, debug symbols, and lines
	python3 -m pip install \
		--verbose \
		--config-settings cmake.args='-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_CXX_FLAGS=-g3;-DCMAKE_C_FLAGS=-g3' \
		--config-settings cmake.verbose=true \
		--config-settings logging.level=INFO \
		--config-settings install.strip=false \
		--editable .

build.openblas: ${submodules} update-pip ## Build ggml-python with openblas support
	python3 -m pip install \
		--verbose \
		--config-settings cmake.args='-DGGML_OPENBLAS=On' \
		--editable .

build.cuda: ${submodules} update-pip ## Build ggml-python with cublas / cuda support
	python3 -m pip install \
		--verbose \
		--config-settings cmake.args='-DGGML_CUDA=On' \
		--editable .

build.clblast: ${submodules} update-pip ## Build ggml-python with clblast / opencl support
	python3 -m pip install \
		--verbose \
		--config-settings cmake.args='-DGGML_CLBLAST=On' \
		--editable .

sdist: ## Build source distribution
	python3 -m build --sdist

deploy: ## Deploy to pypi
	twine upload dist/*

test: ## Run tests
	python3 -m pytest

test.gdb: ## Run tests with gdb
	gdb -ex "set pagination off" -ex r -ex "bt 5" --args python -m pytest -s -vvvv

docs: ## Build documentation using mkdocs and serve it
	mkdocs serve

clean: ## Clean build artifacts
	- rm -rf build
	- rm -rf dist
	- rm ggml/*.so
	- rm ggml/*.dll
	- rm ggml/*.dylib
	- rm ${submodules}/*.so
	- rm ${submodules}/*.dll
	- rm ${submodules}/*.dylib
	- cd ${submodules} && make clean

help: ## Prints help menu
	@grep -E '^[\.a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: \
	all \
	build \
	build.debug \
	build.openblas \
	build.cuda \
	build.clblast \
	sdist \
	deploy \
	test \
	test.gdb \
	docs \
	clean \
	help