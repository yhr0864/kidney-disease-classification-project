#create local packages: KDClassifier, so that you can directly import it from anywhere

import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0" #project version

REPO_NAME = "kidney-disease-classification-project"
AUTHOR_USER_NAME = "yhr0864"
SRC_REPO = "KDClassifier"
AUTHOR_EMAIL = "yuhaoran0864@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"http://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_url={
        "Bug Tracker": f"http://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)