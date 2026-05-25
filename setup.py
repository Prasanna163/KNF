from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='knf',
    version='1.0.8',
    description='Automated Descriptor Engine for SNCI, SCDI, and 9D KNF',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Prasanna Kulkarni',
    license='MIT',
    url='https://github.com/Prasanna163/KNF',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'nciforge=knf_core.main:main',
            'knf=knf_core.main:main',
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'rich',
        'psutil',
        'rdkit',
    ],
    extras_require={
        'torch-nci': ['torch'],
        'plots': ['matplotlib'],
        'full': ['torch', 'matplotlib'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

