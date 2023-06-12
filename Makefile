submodules = vendor/ggml

all: build

${submodules}:
	git submodule update --init --recursive

update-pip:
	python3 -m pip install --upgrade pip

build: ${submodules} update-pip
	python3 -m pip install --verbose --editable .

build.openblas: ${submodules} update-pip
	CMAKE_ARGS="-DGGML_OPENBLAS=On" python3 -m pip install --verbose --editable .

build.cublas: ${submodules} update-pip
	CMAKE_ARGS="-DGGML_CUBLAS=On" python3 -m pip install --verbose --editable .

build.clblast: ${submodules} update-pip
	CMAKE_ARGS="-DGGML_CLBLAST=On" python3 -m pip install --verbose --editable .

sdist:
	python3 -m build --sdist

deploy:
	twine upload dist/*

test:
	python3 -m pytest

clean:
	- rm -rf _skbuild
	- rm -rf dist
	- rm ggml/*.so
	- rm ggml/*.dll
	- rm ggml/*.dylib
	- rm ${submodules}/*.so
	- rm ${submodules}/*.dll
	- rm ${submodules}/*.dylib
	- cd ${submodules} && make clean

.PHONY: all update-pip build build.openblas build.cuda sdist deploy test clean