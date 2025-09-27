import os
from setuptools import setup, find_packages

# Function to read requirements from a file
def read_requirements(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]

# Core dependencies
install_requires = read_requirements('requirements.txt')

# Development dependencies
dev_requires = read_requirements('dev-requirements.txt')

setup(
    name="hpo-engine",
    version="2.0.0",
    author="Jules",
    author_email="jules@example.com",
    description="A professional Hyperparameter Optimization (HPO) engine.",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jules-agent/hpo-engine",
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=["tests", "docs"]),
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    include_package_data=True,
)