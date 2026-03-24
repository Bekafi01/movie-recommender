from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

AUTHOR_NAME = "Beka"
LIST_OF_REQUIREMENTS = [
    "streamlit>=1.32,<2",
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "requests>=2.31",
]

setup(
    name='movie-recommender',
    version='0.1.0',
    author=AUTHOR_NAME,
    description='A simple movie recommender system built with Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=LIST_OF_REQUIREMENTS,
    python_requires='>=3.8',
    )
