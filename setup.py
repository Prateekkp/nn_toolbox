from setuptools import setup, find_packages

setup(
    name="nntoolbox",
    version="0.1.0",
    description="Neural Network Learning Toolbox (Streamlit)",
    author="Prateek Kumar Prasad",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "numpy",
        "pandas",
        "opencv-python",
        "plotly",
        "streamlit-webrtc",
        "av",
    ],
    entry_points={
        "console_scripts": [
            "nntoolbox = nntoolbox.cli:main"
        ]
    },
)