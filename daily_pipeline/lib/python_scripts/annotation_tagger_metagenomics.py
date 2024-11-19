import os
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import requests
import pickle
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForTokenClassification
import argparse
import onnxruntime as ort
# from entity_linker import map_to_url, map_terms_reverse
from entity_tagger import process_each_article, batch_annotate_sentences, extract_annotation, format_output_annotations, SECTIONS_MAP
import torch.nn as nn


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch


PROVIDER = "Metagenomics"

# Define the custom ClassificationModel class
class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use pooler_output or the [CLS] token from the last_hidden_state
        embeddings = (
            outputs.pooler_output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None
            else outputs.last_hidden_state[:, 0, :]
        )
        logits = self.classifier(embeddings)
        return logits, embeddings

def calibrate_logits(logits, temperature=1.0):
    """
    Adjust logits using temperature scaling for better-calibrated probabilities.
    """
    logits = logits / temperature  # Scale logits
    probabilities = torch.softmax(logits, dim=1)  # Calculate probabilities
    return probabilities


def classify_sentence(sentence, temperature=1.0):
    # Tokenize the input sentence
    encoding = tokenizer(
        sentence,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # Predict using the model
    article_model.eval()
    with torch.no_grad():
        outputs = article_model(input_ids, attention_mask)

    # Extract logits from model output
    if isinstance(outputs, tuple):
        logits = outputs[0]  # Extract logits if outputs is a tuple
    else:
        logits = outputs  # Use directly if outputs is not a tuple

    # Apply temperature scaling
    probabilities = calibrate_logits(logits, temperature).cpu().numpy().flatten()

    # Get predicted class and decode label
    predicted_class = probabilities.argmax()  # Class index with the highest probability
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    predicted_probability = probabilities[predicted_class]

    return predicted_label, predicted_probability



# def classify_sentence(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     article_model.eval()
#     with torch.no_grad():
#         outputs = article_model(**inputs)
#     logits = outputs.logits
#
#     # Calculate probabilities from logits
#     probabilities = softmax(logits, dim=1)
#
#     # Get the predicted class and its probability
#     predicted_class = torch.argmax(logits, dim=1).item()
#     predicted_label = label_encoder.inverse_transform([predicted_class])[0]
#     predicted_probability = probabilities[0, predicted_class].item()
#
#     return predicted_label, predicted_probability


def normalize_tag(trm, entity):
    """
    Normalize terms to generate a URI based on the entity type.
    Returns "#" if no match is found.
    """
    # Default URL to indicate no match
    ourl = "#"

    try:
        # Skip single-character terms
        if len(trm) <= 1:
            return ourl

        # Handle general cases for non-'primer' and non-'kit' entities
        if entity not in ['primer', 'kit']:
            url = f'https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate?propertyValue={trm}'
            response = requests.get(url)

            # Check for a successful response
            if response.status_code == 200:
                try:
                    rjson = response.json()
                    if rjson:
                        rslt = rjson[0]
                        lnk = rslt['_links']['olslinks'][0]
                        gurl = lnk.get('semanticTag', [])

                        if 'http://purl.obolibrary.org/obo/' in gurl:
                            db_id = urlparse(gurl).path.split('/')[2:3][0]
                            db = db_id.split('_')[0]
                            ourl = f'https://www.ebi.ac.uk/ols/ontologies/{db}/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F{db_id}'
                        else:
                            ourl = gurl
                except ValueError as e:
                    print(f"{trm}: error - Failed to parse JSON - {e}")
            else:
                print(f"{trm}: error - HTTP status code {response.status_code}")

        # Handle 'kit' entity
        elif entity == 'kit':
            trm = trm.replace(' ', '-')
            ourl = f'https://www.biocompare.com/General-Search/?search={trm}'

        # Handle 'primer' entity
        elif entity == 'primer':
            csvf = Path('xrf/probebase/primer_probebase.csv')
            if csvf.exists():
                prmdf = pd.read_csv(csvf)
                for p, prm in enumerate(prmdf['NAME']):
                    if (prm in trm) or (trm == prm):
                        ourl = list(prmdf['URL'])[p]
                        break
    except Exception as e:
        print(f"{trm}: error - {e}")

    return ourl

def generate_tags(all_annotations):
    """
    Generate tags for each annotation in all_annotations using normalize_tag for term normalization and URL generation.
    Each annotation will have 'name' and 'uri' fields in the 'tags' list.
    """
    output_annotations = []

    # Generate tags for each annotation
    for annotation in all_annotations:
        entity_type = annotation['type']
        term = annotation['exact']

        # Normalize the term to get the URI using normalize_tag
        uri = normalize_tag(term, entity_type)

        # Add the annotation with tags, using "#" if no URI was found
        output_annotations.append({
            "type": entity_type,
            "position": annotation["position"],
            "prefix": annotation["prefix"],
            "exact": term,
            "section": annotation["section"],
            "postfix": annotation["postfix"],
            "tags": [
                {
                    "name": term,
                    "uri": uri if uri != "#" else "#"  # Use "#" if no valid URI is found
                }
            ]
        })

    return output_annotations

# Main function for processing article and generating JSONs

def process_article_generate_jsons(article_data):
    article_type = article_data.get("article_type", "").lower()
    if "research" not in article_type:
        # print(article_type)
        return None, None  # Skip non-research articles

    open_status = article_data.get("open_status", "")
    if open_status not in ["O", "OA"]:
        return None, None  # Skip non OA articles

    pmcid = article_data.get("article_ids", {}).get("pmcid")
    ft_id = article_data.get("article_ids", {}).get("archive") or article_data.get("article_ids", {}).get("manuscript")

    if not pmcid and not ft_id:
        return None, None  # Skip article if no pmcid or ft_id

    # Retrieve and join sentences from the ABSTRACT section
    abstract_section = article_data.get("sections", {}).get("ABSTRACT", [])
    # Ensure each item in abstract_section is a string by extracting the 'text' key
    abstract_text_list = [sentence["text"] for sentence in abstract_section if "text" in sentence]
    # Join the abstract text
    abstract_text = " ".join(abstract_text_list)

    # Classify the article based on the abstract
    predicted_label, proba = classify_sentence(abstract_text)
    print([article_type, open_status, pmcid, predicted_label, proba])
    if predicted_label != "metagenomics" or (predicted_label == "metagenomics" and proba < 0.99):
        return None, None  # Skip NER tagging if label is "other"

    all_annotations = []
    # print([article_type, open_status, pmcid, predicted_label, proba])
    # Loop through each NER model for the specified sections
    # for metagenomic_model in ner_models:
    #     # print(model)
    #     for section_key, sentences in article_data.get("sections", {}).items():
    #         if section_key not in ["INTRO", "METHODS", "RESULTS", "DISCUSS"]:
    #             continue  # Skip all sections except the specified ones
    #
    #         section = SECTIONS_MAP.get(section_key, "Other")
    #         # Pass the NER model and parallel flag to batch_annotate_sentences
    #         batch_annotations = batch_annotate_sentences(sentences, section, ner_model=metagenomic_model, extract_annotation_fn=extract_annotation)
    #         if not batch_annotations:
    #             continue
    #
    #         all_annotations.extend(batch_annotations)
    #
    # # Generate tags for annotations - this includes grounded terms and grounded codes.
    # all_linked_annotations = generate_tags(all_annotations)
    # # Format matched and unmatched JSON structures
    # match_json, non_match_json = format_output_annotations(all_linked_annotations, pmcid=pmcid, ft_id=ft_id)
    #
    # # Return None if both JSONs are empty or have empty 'anns' lists
    # if not match_json["anns"] and not non_match_json["anns"]:
    #     return None, None
    #
    # return match_json, non_match_json






# Main entry point with updated argument parsing
if __name__ == '__main__':
        # Load environment variables
        # load_dotenv('/home/stirunag/work/github/CAPITAL/daily_pipeline/.env_paths')
        ml_model_path = '/home/stirunag/work/github/CAPITAL/model/'

        if not ml_model_path:
            raise ValueError("Environment variable 'MODEL_PATH_QUANTIZED' not found.")

        # Define paths
        article_model_path = ml_model_path + 'article_classifier/best_contrastive_binary_model_absurd-sweep-98.pth'
        article_label_encoder_path = ml_model_path + 'article_classifier/best_label_encoder_absurd-sweep-98.pkl'
        pretrained_model_name = "bioformers/bioformer-8L"
        metagenomic_paths = [
            ml_model_path + f'metagenomics/metagenomic-set-{i}_quantised' for i in range(1, 6)
        ]

        # Set ONNX runtime session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1  # Limit to a single thread
        session_options.inter_op_num_threads = 1  # Limit to a single thread

        # Function to load an ONNX NER model
        def load_ner_model(model_path, session_options):
            try:
                model = ORTModelForTokenClassification.from_pretrained(
                    model_path, file_name="model_quantized.onnx", session_options=session_options
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, model_max_length=512, batch_size=4, truncation=True
                )
                return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="max")
            except Exception as e:
                raise RuntimeError(f"Failed to load NER model from path {model_path}: {str(e)}")

        # Load all NER models
        try:
            ner_models = [load_ner_model(path, session_options) for path in metagenomic_paths]
            print("All NER models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading NER models: {str(e)}")

        # Load the article classifier model
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            # article_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
            encoder = AutoModel.from_pretrained(pretrained_model_name)
            article_model = ClassificationModel(encoder=encoder, num_classes=2)
            article_model.load_state_dict(torch.load(article_model_path, map_location=torch.device('cpu')), strict=False)
            with open(article_label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print("Article classifier model and label encoder loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load article classifier or label encoder: {str(e)}")

        # Define paths
        input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch_2024_10_28_0.json.gz"  # Replace with your actual input file path
        output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/metagenomics/"  # Replace with your actual output directory path

        # Check paths
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Process articles
        process_each_article(
            input_file=input_path,
            output_dir=output_path,
            process_article_json_fn=process_article_generate_jsons,
        )



    # parser = argparse.ArgumentParser(
    #     description='Process section-tagged XML files and output annotations in JSON format.')
    # parser.add_argument('--input', help='Input directory with XML or GZ files', required=True)
    # parser.add_argument('--output', help='Output directory for JSON files', required=True)
    # parser.add_argument('--model_path', help='Path to the quantized model directory', required=True)
    # args = parser.parse_args()
    #
    # # Check that input is a file and output is a directory
    # if not os.path.isfile(args.input):
    #     raise ValueError(f"Expected a file for --input, but got: {args.input}")
    # if not os.path.isdir(args.output):
    #     raise ValueError(f"Expected a directory for --output, but got: {args.output}")
    #
    #
    # # Ensure 'no_matches' directory exists
    # no_match_dir = os.path.join(args.output, "no_matches")
    # os.makedirs(no_match_dir, exist_ok=True)
    # no_match_file_path = os.path.join(no_match_dir, "patch_no_match.json")
    #
    # # Initialize NER model using the provided model path
    #
    # model_path_quantised = args.model_path
    #
    # print("Loading NER model and tokenizer loaded from "+ model_path_quantised)
    # model_quantized = ORTModelForTokenClassification.from_pretrained(
    #     model_path_quantised, file_name="model_quantized.onnx", session_options=session_options)
    # tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised,
    #                                 model_max_length=512, batch_size=4,truncation=True)
    # ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized,
    #                          aggregation_strategy="max")
    # print("NER model and tokenizer loaded successfully.")
    #
    #
    #
    # # Now call process_each_article with input and output directories
    # process_each_article(args.input, args.output)



