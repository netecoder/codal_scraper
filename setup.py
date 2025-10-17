"""
Setup script for Codal Scraper package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="codal-scraper",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive Python package for scraping and analyzing data from Codal.ir",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/codal-scraper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "openpyxl>=3.0.0",
        "jdatetime>=4.1.0",
        "numpy>=1.23.0",
    ],
    extras_require={
        "parquet": ["pyarrow>=10.0.0"],
        "async": [
            "crawlee[playwright]>=0.3.0",
            "playwright>=1.40.0",
        ],
        "network": [
            "networkx>=3.0",
            "pyvis>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ],
        "all": [
            "pyarrow>=10.0.0",
            "crawlee[playwright]>=0.3.0",
            "playwright>=1.40.0",
            "networkx>=3.0",
            "pyvis>=0.3.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "codal-scraper=codal_scraper.cli:main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)