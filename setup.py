from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('README.md', 'r') as fh:
    long_description = fh.read()

extension_modules = [
    # backend.activations模块
    Pybind11Extension(
        'classicML/backend/cc/activations',
        sorted(glob('classicML/backend/cc/activations.cc')),
        include_dirs=[
            '/usr/local/include/eigen3',  # /path/to/eigen3/download
        ],
        language='c++',
    ),
    # backend.kernels模块
    Pybind11Extension(
        'classicML/backend/cc/kernels',
        sorted(glob('classicML/backend/cc/*.cc')),
        include_dirs=[
            '/usr/local/include/eigen3',
        ],
        language='c++',
    ),
    # backend.losses模块
    Pybind11Extension(
        'classicML/backend/cc/losses',
        sorted(glob('classicML/backend/cc/losses.cc')),
        include_dirs=[
            '/usr/local/include/eigen3',
        ],
        language='c++',
    ),
    # backend.metrics模块
    Pybind11Extension(
        'classicML/backend/cc/metrics',
        sorted(glob('classicML/backend/cc/metrics.cc')),
        include_dirs=[
            '/usr/local/include/eigen3',
        ],
        language='c++',
    ),
    # backend.ops模块
    Pybind11Extension(
        'classicML/backend/cc/ops',
        sorted(glob('classicML/backend/cc/*.cc')),
        include_dirs=[
            '/usr/local/include/eigen3',
        ],
        language='c++',
    )
]

setup(
    name='classicML',
    version='0.6b2',
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
        'h5py>=2.10.0',
        'matplotlib>=3.3.2, <=3.3.4',
        'numpy>=1.19.2, <=1.19.4',
        'pandas>=1.1.3, <=1.1.4',
        'psutil>=5.7.2',
    ],
    python_requires='>=3.6',
)
