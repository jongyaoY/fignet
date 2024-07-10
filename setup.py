from setuptools import find_packages, setup

setup(
    name="fignet",
    packages=[
        package
        for package in find_packages()
        if package.startswith("fignet") or package.startswith("rigid_fall")
    ],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    description="FigNet: Face interaction graph networks",
    author="Zongyao Yi",
    url="https://github.com/jongyaoY/fignet.git",
    author_email="zongyao.yi@dfik.de",
    license="MIT License",
    version="0.0.1",
    python_requires=">=3.8",
)
