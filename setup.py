from setuptools import setup, find_packages

setup(
    name="kavian",
    version="0.2.0",
    author="Adam Torres Encarnacion",
    author_email="art5809@psu.edu",
    description="Python statistical modeling toolkit with built-in Pandas and Scikit-Learn integration",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AdamPSU/Kavian",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'rich', 'pytest', 'statsmodels', 'scipy'
    ],
)