from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="parser-shadai",
    version="0.1.7",
    author="Shadai",
    author_email="angie@shadai.ai",
    description="A Python package for parsing PDFs and images using various LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shadai-ai/parser-shadai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=[
        "google-genai==1.38.0",
        "anthropic>=0.68.0",
        "openai>=1.109.1",
        "PyPDF2>=3.0.1",
        "pdf2image>=1.17.0",
        "Pillow>=11.3.0",
        "python-dotenv>=1.1.1",
        "requests>=2.32.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
)
