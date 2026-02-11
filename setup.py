from setuptools import setup, find_packages

setup(
    name="kd-anomalynet",
    version="0.1.0",
    description="Knowledge Distillation for Lightweight Anomaly Detection",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "pyod>=1.0.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "matplotlib>=3.5.0"],
    },
    entry_points={
        "console_scripts": [
            "kd-anomaly=experiments.run_benchmark:main",
        ],
    },
)
