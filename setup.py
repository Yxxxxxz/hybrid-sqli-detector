from setuptools import setup, find_packages

setup(
    name="hybrid-sqli-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib"
    ],
    author="Nithima",
    description="Hybrid SQL Injection Detection using Signature + RandomForest",
    python_requires=">=3.8",
)