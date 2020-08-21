import setuptools

with open("README.md", "r") as read_me:
    long_description = read_me

setuptools.setup(
    name="HLRL",
    version="0.0.1",
    description="Reinforcement learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chainso",
    url="https://github.com/Chainso/HLRL",
    python_requires=">=3.6",
    packages=setuptools.find_packages(),
    install_requires=[
        "pytorch>=1.1",
        "Pillow>=7.0.0",
        "mss>=6.0.0"
    ]
)