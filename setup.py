import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="classicML",
    version="0.3rc2",
    author="Steve Sun",
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
        'numpy>=1.18.4',
        'pandas>=1.0.3',
        'matplotlib>=3.2.1'
    ],
)
