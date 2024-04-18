
# NER Model Training

This repository contains Python code for training Named Entity Recognition (NER) models using Hugging Face transformers and W&B for experiment tracking.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/ML4LitS/CAPITAL.git
cd CAPITAL
cd tagger
pip install -r requirements.txt
```

### Environment Variables

To run this code, you will need to set up the following environment variables:

- `HF_TOKEN`: A Hugging Face API token for accessing model repositories and datasets. [See how to obtain a Hugging Face API key here](https://huggingface.co/docs/hub/security-tokens).
- `WANDB_API_KEY`: An API key for W&B to track and log your experiments. [Find your W&B API key here](https://wandb.ai/authorize).

Set these variables in your environment:

```bash
export HF_TOKEN='your_huggingface_token_here'
export WANDB_API_KEY='your_wandb_api_key_here'
```

### Data Preparation

Your input data must be in a tab-separated format with tokens and their corresponding NER tags. Provide paths for your train, dev, and test datasets by editing the `data_folder` variable in the script:

```python
data_folder = "/path/to/your/input/data/"
```

### Model Training

Configure your model and training settings by editing the parameters in the `train_model` function call within the script. Once configured, you can start the training process by running:

```bash
python train_ner.py
```

The model, its checkpoints, and the final outputs will be saved to the path specified in `model_save_path`.

### Outputs

The script will output:
- Trained model files in the specified `model_save_path`.
- Logs and metrics to your W&B dashboard, facilitating the tracking of training progress and comparison of different runs.

## Using the Code

To use this script for training your own NER models, ensure that you have prepared your data as described, set up the required environment variables, and configured the training settings according to your needs.

For more detailed control over training parameters and for conducting hyperparameter sweeps, refer to the W&B sweep configuration set in the `sweep_config` variable.

## Support

For support using this codebase, please open an issue in the GitHub repository or contact us directly.
