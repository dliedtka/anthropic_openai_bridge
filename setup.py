from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anthropic-openai-bridge",
    version="0.1.0",
    author="David",
    author_email="noreply@example.com",
    description="A Python library that provides an Anthropic Messages API-compatible interface while internally transforming requests to the OpenAI ChatCompletion API format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/anthropic-openai-bridge",
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="anthropic openai api llm chatgpt claude bridge adapter",
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/example/anthropic-openai-bridge/issues",
        "Source": "https://github.com/example/anthropic-openai-bridge",
        "Documentation": "https://github.com/example/anthropic-openai-bridge/blob/main/README.md",
    },
)