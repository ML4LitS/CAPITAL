import os
import shutil
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel, load_metric
import wandb
from huggingface_hub import HfFolder
from wandb.integration.sb3 import WandbCallback

# Initialize the metric
metric = load_metric("seqeval")

#wandb.init(project="epmca_ner_project", entity="tsantosh7", config=hyperparams)

def cleanup_checkpoints(output_dir, keep_last=True, best_model_dir=None, last_model_dir=None):
    """
    Deletes unnecessary model checkpoints created during training.
    Keeps the best model directory and optionally the last model directory.

    :param output_dir: Base directory where the checkpoints are saved.
    :param keep_last: Whether to keep the last checkpoint.
    :param best_model_dir: Directory of the best model checkpoint.
    :param last_model_dir: Directory of the last model checkpoint.
    """
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint"):
            # Check if this path is not the one we want to keep
            if item_path != best_model_dir and (not keep_last or item_path != last_model_dir):
                shutil.rmtree(item_path)

                
# Main training function
def train_model(data_folder, model_save_path,  model_checkpoint):

        # Define and call convert_IOB_transformer, get_token_ner_tags functions
    def convert_IOB_transformer(test_list, pattern):
        new_list = []
        sub_list = []
        for i in test_list:

            if i != pattern:
                sub_list.append(i)
            else:
                new_list.append(sub_list)
                sub_list = []

        return new_list

    def get_token_ner_tags(df_, split_name, label2id_):
        ner_tag_list_ = df_['ner_tags'].map(label2id_).fillna(
            '#*#*#*#*#*#*#*#*').tolist()  # convert the list to a pandas series temporarily before mapping
        token_list_ = df_['tokens'].tolist()

        token_list = convert_IOB_transformer(test_list=token_list_, pattern='')
        ner_tag_list = convert_IOB_transformer(test_list=ner_tag_list_, pattern='#*#*#*#*#*#*#*#*')

        df = pd.DataFrame(list(zip(token_list, ner_tag_list)),
                          columns=['tokens', 'ner_tags'])

        # df.to_csv(path_+'GP-DS-OG-CD-Santosh/'+split_name+'_formatted.tsv', index=None, sep ='\t', header=None)

        return token_list, ner_tag_list, df

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    def tokenize_and_align_labels(examples, device):
        task = "ner"
        label_all_tokens = True
        tokenized_inputs = tokenizer(examples["tokens"], max_length=512, truncation=True, padding="max_length", is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        # labels = torch.tensor(labels).to(dtype=torch.int64)
        # tokenized_inputs["labels"] = labels
        # return tokenized_inputs
        labels = torch.tensor(labels).to(dtype=torch.int64).to(device)  # Move labels to the specified device
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Retrieve HF token from environment and authenticate
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        HfFolder.save_token(hf_token)


    # Load and prepare dataset
    train = pd.read_csv(os.path.join(data_folder, 'train.csv'), sep='\t', names=['tokens', 'ner_tags'], skip_blank_lines=False, na_filter=False)
    # Repeat for dev and test datasets...
    dev = pd.read_csv(os.path.join(data_folder, 'dev.csv'), sep='\t', names=['tokens', 'ner_tags'], skip_blank_lines=False, na_filter=False)
    test = pd.read_csv(os.path.join(data_folder, 'test.csv'), sep='\t', names=['tokens', 'ner_tags'], skip_blank_lines=False, na_filter=False)

      # Dataset processing
    label_list_ = train['ner_tags'].dropna().unique().tolist()
    label_list = [x for x in label_list_ if x]
    id2label = {idx: tag for idx, tag in enumerate(label_list)}
    label2id = {tag: idx for idx, tag in enumerate(label_list)}
    
    dev_tokens, dev_tags, dev_df = get_token_ner_tags(df_=dev, split_name='dev', label2id_=label2id)
    test_tokens, test_tags, test_df = get_token_ner_tags(df_=test, split_name='test', label2id_=label2id)
    train_tokens, train_tags, train_df = get_token_ner_tags(df_=train, split_name='train', label2id_= label2id)

    trds = Dataset.from_pandas(train_df)#, features=features)
    vds = Dataset.from_pandas(dev_df)#, features=features)
    tds = Dataset.from_pandas(test_df)#, features=features)

    ds = DatasetDict()

    ds['train'] = trds
    ds['validation'] = vds
    ds['test'] = tds
    
    # Model initialization
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    if torch.cuda.is_available():
        device = 'cuda'
        model.to(device)
    else:
        device = 'cpu'
        assert torch.cuda.is_available() == True

    # tokenized_datasets = ds.map(tokenize_and_align_labels, batched=True)
        # Apply the tokenize_and_align_labels function to the datasets
    tokenized_datasets = ds.map(lambda x: tokenize_and_align_labels(x, device), batched=True)

    # Read hyperparameters
    with wandb.init() as run:
        hyperparams = run.config
        # Convert the wandb Config object to a standard dictionary
        hyperparams_dict = dict(hyperparams)
        wandb.log({"hyperparams": hyperparams_dict})

        # Dynamic path for saving the model for this run
        model_save_path = os.path.join("models", "run-" + run.id)
        os.makedirs(model_save_path, exist_ok=True)

        # Training arguments and trainer setup
        training_args = TrainingArguments(
                output_dir=model_save_path,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=hyperparams.learning_rate,
                lr_scheduler_type=hyperparams.lr_scheduler_type,
                per_device_train_batch_size=hyperparams.train_batch_size,
                per_device_eval_batch_size=hyperparams.eval_batch_size,
                num_train_epochs=hyperparams.num_train_epochs,
                weight_decay=hyperparams.weight_decay,
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                logging_dir='./logs',
                report_to="wandb"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics  # Define your compute_metrics function
        )

        # Train the model
        trainer.train()

        # The best model is re-loaded at the end of training if 'load_best_model_at_end' is True
        # Evaluate the best model on the test set
        test_results = trainer.evaluate(tokenized_datasets["test"])
        print("Test Set Results:", test_results)
        # Optionally, you can log test results to wandb
        wandb.log({"test_results": test_results})
        #
        # # Save the model
        # trainer.save_model(model_save_path)
        # tokenizer.save_pretrained(model_save_path)

        # Save the best model (after re-loading it)
        best_model_path = os.path.join(model_save_path, "best_model")
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)

        # Optionally, save the last model of the run under a different path or naming
        last_model_path = os.path.join(model_save_path, "last_model")
        trainer.save_model(last_model_path)
        tokenizer.save_pretrained(last_model_path)

        # Log model paths to WandB
        wandb.log({"best_model_path": best_model_path, "last_model_path": last_model_path})

        # Cleanup intermediate checkpoints
        cleanup_checkpoints(
            output_dir=model_save_path,
            keep_last=True,
            best_model_dir=best_model_path,
            last_model_dir=last_model_path
)

# Entry point for the script
if __name__ == "__main__":

    sweep_config = {
        'method': 'random',  # or 'grid' for exhaustive search
        'metric': {
            'name': 'f1',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [5e-5, 2e-5, 1e-5]
            },
            'train_batch_size': {
                'values': [16]
            },
            'eval_batch_size': {
                'values': [16]
            },
            'num_train_epochs': {
                'values': [10]
            },
            'weight_decay': {
                'values': [0.1, 0.01, 0.001]
            },
            'lr_scheduler_type': {
                'values': ['linear'] # 'cosine', 'polynomial'
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 2
        }
    }

    data_folder = "/nfs/production/literature/santosh_tirunagari/datasets/enriched_IOB/"
    model_save_path = "/nfs/production/literature/santosh_tirunagari/models/enriched_model/"
    pretrained_model = "bioformers/bioformer-8L"

    sweep_id = wandb.sweep(sweep_config, project="ebi_epmca_project", entity="ebi_literature")
    # wandb.agent(sweep_id, train)
    wandb.agent(sweep_id, lambda: train_model(data_folder, model_save_path, pretrained_model))

