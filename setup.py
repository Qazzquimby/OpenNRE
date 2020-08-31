import setuptools

import subprocess

TORCH_SUCCESS_MESSAGE = """Successfully installed pytorch and torchvision CPU version.
If you need the GPU version, please install it manually. https://pytorch.org/get-started/locally/"""

TORCH_FAIL_MESSAGE = """Failed to install pytorch, please install pytorch and torchvision manually
by following the instructions at: https://pytorch.org/get-started/locally/"""

code = 1
try:
    code = subprocess.call(['pip', 'install', 'torch===1.4.0+cpu', 'torchvision===0.5.0+cpu', '-f',
                            'https://download.pytorch.org/whl/torch_stable.html'])
    if code != 0:
        raise EnvironmentError(TORCH_FAIL_MESSAGE)
except EnvironmentError:
    try:
        code = subprocess.call(['pip3', 'install', 'torch===1.4.0+cpu', 'torchvision===0.5.0+cpu', '-f',
                                'https://download.pytorch.org/whl/torch_stable.html'])
        if code != 0:
            raise EnvironmentError(TORCH_FAIL_MESSAGE)
    except EnvironmentError:
        print(TORCH_FAIL_MESSAGE)
if code == 0:
    print(TORCH_SUCCESS_MESSAGE)

with open("README.md", "r") as fh:
    setuptools.setup(
        name='opennre',
        version='0.1',
        author="Tianyu Gao",
        author_email="gaotianyu1350@126.com",
        description="An open source toolkit for relation extraction",
        url="https://github.com/thunlp/opennre",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Linux",
        ],
        setup_requires=['wheel'],
        install_requires=[
            'transformers',
            'pytest',
            'scikit-learn',
            'scipy',
            'nltk'
        ]
    )
