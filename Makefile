all: build

build:
	python3 setup.py develop

sdist:
	python3 setup.py sdist

test:
	pytest

clean:
	- rm -rf _skbuild
	- rm -rf dist
	- rm ggml/*.so
	- rm ggml/*.dll
	- rm ggml/*.dylib
	- cd vendor/ggml && make clean

.PHONY: all build sdist test clean