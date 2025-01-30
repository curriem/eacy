from setuptools import setup, find_packages

setup(
    name="eacy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["hwo_sci_eng"],
    author="Miles Currie",
    author_email="miles.h.currie@nasa.gov",
    description="A simple package to collect and collate HWO YAML files for EACs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/curriem/eacy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
