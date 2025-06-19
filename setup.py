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
    description = "A simplified package to use BERT with MLX",
    author = "Luis Herrera G.",
    author_email = "u202218227@upc.edu.pe",
    url = "https://github.com/LuisHerreraG/easy_bert_mlx",
)
