## Running the Notebooks

1. Check if Conda is installed:
    ```bash
    conda --version
    ```
2. If Conda is not installed, download and install it from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

3. Create and activate the environment and install dependencies:
    ```bash
    conda create -n cs8903 python=3.13.11 -y
    conda activate cs8903
    pip install -r requirements.txt
    ```
4. For some raw data info, refer: https://github.com/knicklabs/lake-malawi/tree/main


## Running the Web Application (Next.js)

To run the web application locally:

1. Execute below in terminal
   ```bash
   cd web-app && npm install && npm run dev
   ```
2. Open [http://localhost:3000](http://localhost:3000) in your browser.

To build and test the application (including Lighthouse audit):

```bash
npm run build
npm run lighthouse
```
