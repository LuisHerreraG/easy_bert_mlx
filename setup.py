from setuptools import setup, find_packages

setup(
    name="easy-bert-mlx",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "mlx",
        "huggingface_hub",
    ],
    description="A simplified package to use BERT with MLX",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/easy_bert_mlx",
)
