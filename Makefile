all: build

build:
	python3 setup.py develop

test:
	pytest

clean:
	- rm -rf _skbuild
	- rm ggml/*.so
	- rm ggml/*.dll
	- rm ggml/*.dylib
	- cd vendor/ggml && make clean

.PHONY: all build test clean