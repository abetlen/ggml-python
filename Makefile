all: build

build:
	python3 setup.py develop

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
	- rm vendor/ggml/*.{so,dll,dylib}
	- cd vendor/ggml && make clean

.PHONY: all build sdist deploy test clean