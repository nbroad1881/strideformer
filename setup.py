from setuptools import find_packages, setup


LONG_DESCRIPTION = """
A package that includes an implementation of a two-transformer
setup to classify long documents. The first stage is a 
sentence-transformer which embeds chunks of the text. The second 
stage is a generic transformer used to classify the documents
based on a sequence of embeddings from the first model. 
Training scripts included.
"""

REQUIRED_PKGS = ["transformers>=4.21.0"]

QUALITY_REQUIRE = ["black", "flake8", "isort", "tabulate"]
TRAIN_REQUIRE = ["hydra-core>=1.2.0"]

EXTRAS_REQUIRE = {"quality": QUALITY_REQUIRE, "train": TRAIN_REQUIRE}


def combine_requirements(base_keys):
    return list(set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements([k for k in EXTRAS_REQUIRE])


setup(
    name="strideformer",
    version="0.1.2",
    description="Package to use a two-stage transformer for long-document classification",
    long_description=LONG_DESCRIPTION,
    author="Nicholas Broad",
    author_email="nicholas@huggingface.co",
    url="https://github.com/nbroad1881/strideformer",
    download_url="https://github.com/nbroad1881/strideformer",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    keywords="nlp, machine learning, transformers",
)