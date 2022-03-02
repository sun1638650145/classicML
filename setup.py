from glob import glob
from platform import system
from unittest.mock import Mock

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

if system() == 'Windows':
    # /path/to/eigen3/download
    EIGEN_DIR = 'c:/vcpkg/installed/x64-windows/include/eigen3'
    # 用于解决在Windows下构建时, 链接错误 error LNK2001: unresolved external symbol
    build_ext.get_export_symbols = Mock(return_value=None)
else:
    # /path/to/eigen3/download
    EIGEN_DIR = '/usr/local/include/eigen3'

with open('README.md', 'r', encoding='utf-8') as fp:
    long_description = fp.read()

extension_modules = [
    # backend._utils模块
    Pybind11Extension(
        'classicML/backend/cc/_utils',
        sorted(glob('classicML/backend/cc/_utils/*.cc')),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.activations模块
    Pybind11Extension(
        'classicML/backend/cc/activations',
        sorted(glob('classicML/backend/cc/activations/*.cc')),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.callbacks模块
    Pybind11Extension(
        'classicML/backend/cc/callbacks',
        sorted(glob('classicML/backend/cc/callbacks/*.cc')),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.initializers模块
    Pybind11Extension(
        'classicML/backend/cc/initializers',
        sorted(glob('classicML/backend/cc/initializers/*.cc') + ['classicML/backend/cc/matrix_op.cc']),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.kernels模块
    Pybind11Extension(
        'classicML/backend/cc/kernels',
        sorted(glob('classicML/backend/cc/kernels/*.cc') + ['classicML/backend/cc/matrix_op.cc']),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.losses模块
    Pybind11Extension(
        'classicML/backend/cc/losses',
        sorted(glob('classicML/backend/cc/losses/*.cc')),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.metrics模块
    Pybind11Extension(
        'classicML/backend/cc/metrics',
        sorted(glob('classicML/backend/cc/metrics/*.cc')),
        include_dirs=[EIGEN_DIR],
        language='c++',
    ),
    # backend.ops模块
    Pybind11Extension(
        'classicML/backend/cc/ops',
        sorted(glob('classicML/backend/cc/ops/*.cc') + ['classicML/backend/cc/matrix_op.cc']),
        include_dirs=[EIGEN_DIR],
        language='c++',
    )
]

setup(
    name='classicML',
    version='0.8rc0',
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    license='Apache Software License',
    cmdclass={'build_ext': build_ext},
    install_requires=[
        'h5py>=3.4.0, <=3.6.0',
        'matplotlib>=3.5.0, <=3.5.1',
        'numpy>=1.21.0, <=1.22.2',
        'pandas>=1.3.4, <=1.4.1',
        'psutil>=5.7.2, <=5.9.0',
    ],
    python_requires='>=3.7',
)
