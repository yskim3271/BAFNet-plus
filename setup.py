from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bafnet-plus",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Semi-Real-Time Speech Enhancement with PrimeKnet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/BAFNet-plus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0",
        "librosa==0.10.1",
        "nlptutti==0.0.0.8",
        "numpy>=1.26.0",
        "pesq==0.0.4",
        "tensorboard>=2.15.0",
        "scipy",
        "datasets",
        "psutil",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
    },
)
