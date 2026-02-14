from setuptools import setup, find_packages

setup(
    name='knf-core',
    version='0.1.0',
    description='Automated Descriptor Engine for SNCI, SCDI, and 9D KNF',
    author='Prasanna Kulkarni',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'knf=knf_core.main:main',
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        # 'rdkit', # Assuming rdkit is pre-installed in the environment (e.g., conda)
    ],
    python_requires='>=3.8',
)
