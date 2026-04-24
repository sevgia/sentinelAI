from setuptools import setup, find_packages

setup(
    name="sentinel-ai",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.2.0,<2.3",
        "torchvision>=0.17.0,<0.18",
        "torchaudio>=2.2.0,<2.3",
        "opacus>=1.4.0,<1.5.0",
        "fairlearn",
        "mlflow",
        "scikit-learn",
        "pandas",
        "numpy>=1.26.4,<2",
    ],
)