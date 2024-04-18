# Generate Datasets

We will use Europe PMC APIs to download the datasets
## Prerequisites

Before you begin, ensure you have Python installed on your system. Python 3.6 or later is recommended.

## Setting Up the Environment

1. **Create a Python Virtual Environment:**

    Run the following command to create a new virtual environment named `falconframes_env` in your home directory.

    ```bash
    python -m venv ~/falconframes_env
    ```

2. **Activate the Virtual Environment:**

    Activate the virtual environment using the command below:

    ```bash
    source ~/falconframes_env/bin/activate
    ```

    > Note: On Windows, the activation command is different. Use `~/falconframes_env\Scripts\activate`.

## Installing Dependencies

With the virtual environment activated, install the required libraries using `pip`:

```bash
pip install notebook
pip install matplotlib
pip install lxml
pip install ipywidgets
```

## Jupyter Notebook Extensions
Enable the required Jupyter Notebook extensions:

```bash
jupyter nbextension enable --py widgetsnbextension
```

## Installing SciSpacy
Install SciSpacy and its dependencies:

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

## Usage
After installation, you can start using Jupyter Notebook for the FalconFrames project. Ensure the virtual environment is activated whenever you work on the project.

## Deactivating the Environment
When you're done, you can deactivate the virtual environment by running:

```bash
deactivate
```
