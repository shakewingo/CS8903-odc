## 1. Environment Setup

To run the notebooks and the application:

First, check if Conda is installed:
```bash
conda --version
```
If Conda is not installed, download and install it from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

Then create and activate the environment and install dependencies:
```bash
conda create -n cs8903 python=3.13.11 -y
conda activate cs8903
pip install -r requirements.txt
```

## 2. Running the Streamlit App

To run the Streamlit dashboard locally, execute the following command from the project root:

```bash
streamlit run app.py
```
Note that this is not a PWA web applications yet, and will be modified in the future to meet course requirements.
