from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('README.md', 'r') as fh:
    long_description = fh.read()

extension_modules = [
    Pybind11Extension(
        'classicML/backend/cc/ops',
        sorted(glob('classicML/backend/cc/*.cc')),
        include_dirs=[
            '/usr/local/include/eigen3',  # /path/to/eigen3/download 默认为macOS的路径, Ubuntu是/usr/include/eigen3
        ],
        language='c++',
    )
]

setup(
    name='classicML',
    version='0.5b2',
    author='Steve R. Sun',
    license='Apache Software License',
    author_email='s1638650145@gmail.com',
    description='An easy-to-use ML framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sun1638650145/classicML',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib>=3.3.2',
        'numpy>=1.19.2',
        'pandas>=1.1.3',
        'psutil>=5.7.2',
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=extension_modules,
)
