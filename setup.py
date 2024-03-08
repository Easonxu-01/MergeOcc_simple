'''
Author: EASON XU
Date: 2023-10-01 12:30:52
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2023-10-09 06:38:27
Description: 头部注释
FilePath: /UniLiDAR/setup.py
'''
from setuptools import find_packages, setup

import os
import torch
from os import path as osp
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)



if __name__ == '__main__':
    # add_mim_extention()
    setup(
        name='UniLiDAR',
        version='0.0',
        description=("UniLiDAR: bridge the gap between different kind of LiDARs"),
        author='UniLiDAR Contributors',
        author_email='xuzk23@mails.tsinghua.edu.cn',
        keywords='UniLiDAR Perception',
        packages=find_packages(),
        include_package_data=True,
        package_data={'projects.unilidar_plugin.ops': ['*/*.so']},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="Apache License 2.0",

        ext_modules=[
            make_cuda_ext(
                name="occ_pool_ext",
                module="projects.unilidar_plugin.ops.occ_pooling",
                sources=[
                    "src/occ_pool.cpp",
                    "src/occ_pool_cuda.cu",
                ]),
        ],
        cmdclass={'build_ext': BuildExtension})