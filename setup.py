import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read()

setuptools.setup(
    name="predhpc",
    version="0.0.1",
    author="Colleen J. Gillon",
    author_email="c.gillon@imperial.ac.uk",
    description="A package for investigating how predictive activity emerges in the HPC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colleenjg/predhpc",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
