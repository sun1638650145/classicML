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
            '/usr/local/include/eigen3',  # /path/to/eigen3/download
        ],
        language='c++',
    )
]

setup(
    name='classicML',
    version='0.5b4',
    description='An easy-to-use ML framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Steve R. Sun',
    author_email='s1638650145@gmail.com',
    url='https://github.com/sun1638650145/classicML',
    packages=find_packages(),
    ext_modules=extension_modules,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='Apache Software License',
    cmdclass={'build_ext': build_ext},
    install_requires=[
        'matplotlib>=3.3.2',
        'numpy>=1.19.2, <=1.19.4',
        'pandas>=1.1.3',
        'psutil>=5.7.2',
    ],
    python_requires='>=3.6',
)
