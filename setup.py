import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="classicML",
    version="0.5alpha2",
    author="Steve R. Sun",
    license='Apache Software License',
    author_email="s1638650145@gmail.com",
    description="An easy-to-use ML framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sun1638650145/classicML",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.1.3',
        'matplotlib>=3.3.2'
    ],
)
