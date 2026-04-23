from setuptools import setup, find_packages

setup(
    name="ml_model_fraud_detection",
    version="1.0.0",
    description="End-to-end ML pipeline for credit card fraud detection",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": ["pytest>=8.1.0", "pytest-cov>=5.0.0", "httpx>=0.27.0"],
    },
)
