from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
VERSION = "1.1.0"

install_requires = [
    "requests>=2.28.0",
    "pandas>=1.5.0",
    "beautifulsoup4>=4.11.0",
    "jdatetime>=4.1.0",
    "networkx>=3.0",
    "openpyxl>=3.1.0",
    "pyarrow>=14.0.0",
    "aiohttp>=3.9.0",
    "tqdm>=4.65.0",
]

extras = {
    "async": ["crawlee[playwright]>=0.2.0", "playwright>=1.40.0"],
    "viz": ["pyvis>=0.3.0"],
}

setup(
    name="codal_scraper",
    version=VERSION,
    description="High level tools for scraping Codal.ir disclosures.",
    long_description="Codal Scraper provides sync and async clients, caching, and processors for Codal.ir.",
    long_description_content_type="text/plain",
    author="Mohammad Mehdi Pakravan",
    url="https://codal.ir",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
)
