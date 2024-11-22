import os
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import requests
import torch.nn.functional as F
import torch
from markdown_it.common.entities import entities
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForTokenClassification
import argparse
import onnxruntime as ort
from entity_tagger import process_each_article, batch_annotate_sentences, extract_annotation, format_output_annotations, SECTIONS_MAP
# from entity_linker import load_annotations, map_to_url, map_terms_reverse
from entity_linker import EntityLinker
import torch.nn as nn
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
from transformers import AutoModel, AutoTokenizer

# Global variables for models and artifacts
PROVIDER = "Metagenomics"
sim_model = None
sim_tokenizer = None
label_encoder = None
clf = None

# Set ONNX runtime session options
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1  # Limit to a single thread
session_options.inter_op_num_threads = 1  # Limit to a single thread

ENTITY_TYPE_MAP = {
    "sample-material": "Sample-Material",
    "body-site": "Body-Site",
    "host": "Host",
    "state": "State",
    "site": "Site",
    "place": "Place",
    "date": "Date",
    "engineered": "Engineered",
    "ecoregion": "Ecoregion",
    "treatment": "Treatment",
    "kit": "Kit",
    "primer": "Primer",
    "gene": "Gene",
    "ls": "LS",
    "lcm": "LCM",
    "sequencing": "Sequencing"
}


# Helper Functions
def map_entity_type(abbrev):
    """Map abbreviation to full form."""
    return ENTITY_TYPE_MAP.get(abbrev, abbrev.upper())



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


# Load all models and artifacts once
def load_artifacts(base_path):
    """
    Load all necessary models and artifacts into memory.

    Args:
        base_path (str): Path to the folder containing saved models and artifacts.
    """
    global sim_model, sim_tokenizer, label_encoder, clf

    # Paths to each folder
    model_folder = os.path.join(base_path, "hf_model")
    tokenizer_folder = os.path.join(base_path, "hf_tokenizer")
    label_encoder_path = os.path.join(base_path, "label_encoder", "label_encoder.pkl")
    logreg_path = os.path.join(base_path, "logistic_regression", "logistic_regression.pkl")

    # Load Hugging Face model and tokenizer
    sim_model = AutoModel.from_pretrained(model_folder)
    sim_tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)

    # Load LabelEncoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Load Logistic Regression classifier
    with open(logreg_path, 'rb') as f:
        clf = pickle.load(f)

    # print("Artifacts loaded successfully.")

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling function that takes attention masks into account.
    """
    token_embeddings = model_output[0]  # First element: token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def classify_abstract(abstract):
    """
    Classifies an abstract using preloaded models and artifacts.

    Args:
        abstract (str): The abstract to classify.

    Returns:
        dict: A dictionary with predicted class label and probabilities.
    """
    global sim_model, sim_tokenizer, label_encoder, clf

    # Ensure models and artifacts are loaded
    if sim_model is None or sim_tokenizer is None or label_encoder is None or clf is None:
        raise RuntimeError("Artifacts are not loaded. Please call `load_artifacts` first.")

    # Process the input abstract
    sim_model.eval()  # Ensure the model is in evaluation mode
    encoded_input = sim_tokenizer(abstract, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = sim_model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1).cpu().numpy()

    # Predict the class label and probability
    predicted_probabilities = clf.predict_proba(sentence_embedding)[0]
    predicted_label = clf.predict(sentence_embedding)[0]
    class_label = label_encoder.inverse_transform([predicted_label])[0]

    # Return the results
    return class_label, predicted_probabilities.max()

def normalize_tag(trm, entity):
    """
    Normalize terms to generate a URI based on the entity type.
    Returns "#" if no match is found.

    Args:
        trm (str): The term to normalize.
        entity (str): The entity type (e.g., 'primer', 'kit').


    Returns:
        str: The generated URI or '#' if no match is found.
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
            mapped_results = linker.map_terms_reverse(entities=[trm], entity_type='primer')
            if entity in mapped_results and trm in mapped_results[entity]:
                grounded_code, _ = mapped_results[entity][trm]
                ourl = linker.map_to_url(entity_group='Primer', ent_id=grounded_code)

    except Exception as e:
        print(f"{trm}: error - {e}")

    return ourl



# def normalize_tag(trm, entity, primer_df=None):
#     """
#     Normalize terms to generate a URI based on the entity type.
#     Returns "#" if no match is found.
#
#     Args:
#         trm (str): The term to normalize.
#         entity (str): The entity type (e.g., 'primer', 'kit').
#         primer_df (pd.DataFrame, optional): Preloaded primer dictionary as a DataFrame.
#
#     Returns:
#         str: The generated URI or '#' if no match is found.
#     """
#     # Default URL to indicate no match
#     ourl = "#"
#
#     try:
#         # Skip single-character terms
#         if len(trm) <= 1:
#             return ourl
#
#         # Handle general cases for non-'primer' and non-'kit' entities
#         if entity not in ['primer', 'kit']:
#             url = f'https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate?propertyValue={trm}'
#             response = requests.get(url)
#
#             # Check for a successful response
#             if response.status_code == 200:
#                 try:
#                     rjson = response.json()
#                     if rjson:
#                         rslt = rjson[0]
#                         lnk = rslt['_links']['olslinks'][0]
#                         gurl = lnk.get('semanticTag', [])
#
#                         if 'http://purl.obolibrary.org/obo/' in gurl:
#                             db_id = urlparse(gurl).path.split('/')[2:3][0]
#                             db = db_id.split('_')[0]
#                             ourl = f'https://www.ebi.ac.uk/ols/ontologies/{db}/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F{db_id}'
#                         else:
#                             ourl = gurl
#                 except ValueError as e:
#                     print(f"{trm}: error - Failed to parse JSON - {e}")
#             else:
#                 print(f"{trm}: error - HTTP status code {response.status_code}")
#
#         # Handle 'kit' entity
#         elif entity == 'kit':
#             trm = trm.replace(' ', '-')
#             ourl = f'https://www.biocompare.com/General-Search/?search={trm}'
#
#         # Handle 'primer' entity
#         elif entity == 'primer' and primer_df is not None:
#             for p, prm in enumerate(primer_df['NAME']):
#                 if (prm in trm) or (trm == prm):
#                     ourl = list(primer_df['URL'])[p]
#                     break
#
#     except Exception as e:
#         print(f"{trm}: error - {e}")
#
#     return ourl


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
        # uri = normalize_tag(term, entity_type, primer_df=primer_df)
        # Add the annotation with tags, using "#" if no URI was found
        output_annotations.append({
            "type": map_entity_type(entity_type),
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
    predicted_label, proba = classify_abstract(abstract_text)
    if predicted_label != "metagenomics" : #or (predicted_label == "metagenomics" and proba < 0.8)
        return None, None  # Skip NER tagging if label is "other"

    all_annotations = []
    print([article_type, open_status, pmcid, predicted_label, proba])
    # Loop through each NER model for the specified sections
    for metagenomic_model in ner_models:
        # print(model)
        for section_key, sentences in article_data.get("sections", {}).items():
            if section_key not in ["INTRO", "METHODS", "RESULTS", "DISCUSS"]:
                continue  # Skip all sections except the specified ones

            section = SECTIONS_MAP.get(section_key, "Other")
            # Pass the NER model and parallel flag to batch_annotate_sentences
            batch_annotations = batch_annotate_sentences(sentences, section, ner_model=metagenomic_model, extract_annotation_fn=extract_annotation)
            if not batch_annotations:
                continue

            all_annotations.extend(batch_annotations)

    # Generate tags for annotations - this includes grounded terms and grounded codes.
    all_linked_annotations = generate_tags(all_annotations)
    # Format matched and unmatched JSON structures
    match_json, non_match_json = format_output_annotations(all_linked_annotations, pmcid=pmcid, ft_id=ft_id)

    # Return None if both JSONs are empty or have empty 'anns' lists
    if not match_json["anns"] and not non_match_json["anns"]:
        return None, None
    #
    # print([abstract_text, article_type, open_status, pmcid, predicted_label, proba])
    return match_json, non_match_json


# Main entry point with updated argument parsing
if __name__ == '__main__':
        # Load environment variables
        # load_dotenv('/home/stirunag/work/github/CAPITAL/daily_pipeline/.env_paths')
        ml_model_path = '/home/stirunag/work/github/CAPITAL/model/'
        primer_dictionary_path = '/home/stirunag/work/github/CAPITAL/normalisation/dictionary/'
        article_classifier_path = ml_model_path+"article_classifier/"

        # Instantiate the EntityLinker class
        linker = EntityLinker()
        loaded_data = linker.load_annotations(['primer'])

        if not ml_model_path:
            raise ValueError("Environment variable 'MODEL_PATH_QUANTIZED' not found.")


        metagenomic_paths = [
            ml_model_path + f'metagenomics/metagenomic-set-{i}_quantised' for i in range(1, 6)
        ]

        # Load all NER models
        try:
            ner_models = [load_ner_model(path, session_options) for path in metagenomic_paths]
            print("All NER models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading NER models: {str(e)}")

        # Load all NER models
        try:
            load_artifacts(article_classifier_path)
            print("All article classifier models loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading article classifier models: {str(e)}")



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



