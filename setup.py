from setuptools import setup, find_packages

setup(
    name="raggy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "openai",
        "langchain",
        "faiss-cpu",
        "pypdf",
        "flask",
        "flask-socketio",
        "python-socketio",
    ],
    entry_points={
        "console_scripts": [
            "raggy=raggy.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "raggy": ["interfaces/front-end/build/**/*"],
    },
) 