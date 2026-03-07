from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

AUTHOR_NAME = "Beka"
LIST_OF_REQUIREMENTS = ['streamlit']

setup(
    name='movie-recommender',
    version='0.1.0',
    author=AUTHOR_NAME,
    description='A simple movie recommender system built with Python.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['movie_recommender'],
    install_requires=LIST_OF_REQUIREMENTS,
    python_requires='>=3.8',
    )