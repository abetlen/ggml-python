from skbuild import setup

setup(
    name="ggml_python",
    description="Python bindings for ggml",
    long_description_content_type="text/markdown",
    version="0.0.1",
    author="Andrei Betlen",
    author_email="abetlen@gmail.com",
    license="MIT",
    packages=["ggml"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
