from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nciforge',
    version='1.0.9',
    description='Automated Descriptor Engine for SNCI, SCDI, and 9D KNF',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Prasanna Kulkarni',
    license='MIT',
    url='https://github.com/Prasanna163/NCIForge',
    packages=find_packages(),
    py_modules=['nciforge_cli'],
    entry_points={
        'console_scripts': [
            'nciforge=nciforge_cli:main',
            'knf=nciforge_cli:main',
            'geoinit=nciforge_cli:geoinit_main',
            'xtbx=nciforge_xtbx.cli:main',
            'nciforge-api=nciforge_cli:api_main',
        ],
    },
    package_data={
        'nciforge_xtbx': [
            'xtbx_run.sh',
            'xtbg.conf',
            'runtime/README.md',
            'runtime/LICENSES/*',
            'runtime/xtb-win-release/bin/*.exe',
            'runtime/xtb-win-release/lib/*.dll',
            'runtime/xtb-win-release/params/*',
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'rich',
        'psutil',
        'rdkit',
        'typer>=0.12',
    ],
    extras_require={
        'api': ['fastapi', 'uvicorn[standard]', 'python-multipart'],
        'torch-nci': ['torch'],
        'plots': ['matplotlib'],
        'geoinit-benchmark': ['pandas', 'matplotlib'],
        'full': [
            'torch',
            'matplotlib',
            'pandas',
            'fastapi',
            'uvicorn[standard]',
            'python-multipart',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
