submodules = vendor/ggml

all: build

${submodules}:
	git submodule update --init --recursive

build: ${submodules}
	python3 -m pip install --verbose --editable .

build.cuda: ${submodules}
	CMAKE_ARGS="-DGGML_CUBLAS=On" python3 -m pip install --verbose --editable .

sdist:
	python3 setup.py sdist

deploy:
	twine upload dist/*

test:
	pytest

clean:
	- rm -rf _skbuild
	- rm -rf dist
	- rm ggml/*.{so,dll,dylib}
	- rm ${submodules}/*.{so,dll,dylib}
	- cd ${submodules} && make clean

.PHONY: all build sdist deploy test clean