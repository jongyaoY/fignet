# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "readme.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="fignet",
    packages=[package for package in find_packages() if package.startswith("fignet")],
    install_requires=[

    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="FigNet: Face interaction graph networks",
    author="Zongyao Yi",
    url="https://github.com/jongyaoY/fignet.git",
    author_email="zongyao.yi@dfik.de",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
