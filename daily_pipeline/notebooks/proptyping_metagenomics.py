#%%
import json
import gzip
from collections import defaultdict, OrderedDict, Counter
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForTokenClassification
import argparse
from tqdm import tqdm
import onnxruntime as ort
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv
from itertools import islice

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




SECTIONS_MAP = {
    "TITLE": "Title",
    "ABSTRACT": "Abstract",
    "INTRO": "Introduction",
    "METHODS": "Methods",
    "RESULTS": "Results",
    "DISCUSS": "Discussion",
    "CONCL": "Conclusion",
    "CASE": "Case study",
    "ACK_FUND": "Acknowledgments",
    "AUTH_CONT": "Author Contributions",
    "COMP_INT": "Competing Interests",
    "ABBR": "Abbreviations",
    "SUPPL": "Supplementary material",
    "REF": "References",
    "TABLE": "Table",
    "FIGURE": "Figure",
    "DATA SHARING STATEMENT": "Data Availability",
    "APPENDIX": "Appendix",
    "OTHER": "Other"
}



PROVIDER = "Metagenomics"
PARALLEL_ = True


#%%
def get_word_position(sent_id, sentence_text, char_start):
    """
    Calculate the word position based on character start index.
    Returns a string in the format 'sent_id.word_position'.
    """
    words = sentence_text.split()
    current_char = 0
    for idx, word in enumerate(words):
        word_start = sentence_text.find(word, current_char)
        if word_start == char_start:
            return f"{sent_id}.{idx + 1}"
        current_char = word_start + len(word)
    return f"{sent_id}.0"  # Return 0 if position not found


def get_prefix_postfix(sentence_text, char_start, char_end, num_words=3, max_chars=30):
    """
    Extract prefix and postfix based on word positions with constraints:
    - Returns up to `num_words` before and after the target term.
    - Ensures each prefix and postfix does not exceed `max_chars`.
    """
    words = sentence_text.split()
    word_positions = [sentence_text.find(word) for word in words]

    # Identify the word index for the starting character of the entity
    word_index = None
    for idx, start in enumerate(word_positions):
        if start == char_start:
            word_index = idx
            break

    prefix, postfix = "", ""
    if word_index is not None:
        # Extract prefix words up to `num_words` or `max_chars`
        prefix_words = words[max(0, word_index - num_words):word_index]
        prefix = ' '.join(prefix_words)
        if len(prefix) > max_chars:
            prefix = prefix[-max_chars:]  # Truncate to the last `max_chars` characters

        # Extract postfix words up to `num_words` or `max_chars`
        postfix_words = words[word_index + 1:word_index + 1 + num_words]
        postfix = ' '.join(postfix_words)
        if len(postfix) > max_chars:
            postfix = postfix[:max_chars]  # Truncate to the first `max_chars` characters

    return prefix, postfix

# Helper function to batch sentences into chunks of size n
def batch_sentences(iterable, n=4):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def extract_annotation(sentence_id, sentence_text, entity, section):
    term = sentence_text[entity['start']:entity['end']]
    position = get_word_position(sentence_id, sentence_text, entity['start'])
    prefix, postfix = get_prefix_postfix(sentence_text, entity['start'], entity['end'])
    full_entity_type = entity["entity_group"]

    return {
        "type": full_entity_type,
        "position": position,
        "prefix": prefix,
        "exact": term,
        "section": section,
        "postfix": postfix
    }

# Annotate sentences in batches and count entity frequency
def batch_annotate_sentences(sentences, section, ner_model, parallel=PARALLEL_):
    annotations = []

    if parallel:
        # Process in batches of at least 4 sentences
        for sentence_batch in batch_sentences(sentences, n=4):
            batched_text = [s["text"] for s in sentence_batch]
            ner_results = ner_model(batched_text)

            for i, sentence_entities in enumerate(ner_results):
                sentence_id = sentence_batch[i]["sent_id"]
                sentence_text = sentence_batch[i]["text"]
                for entity in sentence_entities:
                    annotations.append(extract_annotation(sentence_id, sentence_text, entity, section))
    else:
        # Process sentences individually
        for sentence in sentences:
            ner_results = ner_model([sentence["text"]])[0]
            sentence_id = sentence["sent_id"]
            sentence_text = sentence["text"]
            for entity in ner_results:
                annotations.append(extract_annotation(sentence_id, sentence_text, entity, section))

    return annotations

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



def format_output_annotations(all_linked_annotations_, pmcid, ft_id):
    """
    Formats output annotations into two JSON structures:
    - 'match_json' for matched annotations
    - 'non_match_json' for unmatched annotations
    """
    match_annotations = []
    non_match_annotations = []

    # Separate annotations based on tags
    for annotation in all_linked_annotations_:
        # Check if the annotation is unmatched (name and uri are '#')
        if annotation["tags"][0]["uri"].endswith("#"):
            non_match_annotations.append(annotation)
        else:
            match_annotations.append(annotation)

    # Construct final JSON outputs
    match_json = OrderedDict()
    non_match_json = OrderedDict()

    # Add pmcid or ft_id to both match and non-match JSONs
    if pmcid:
        match_json["pmcid"] = pmcid
        non_match_json["pmcid"] = pmcid
    elif ft_id:
        match_json["ft_id"] = ft_id
        non_match_json["ft_id"] = ft_id

    # Add provider and anns fields to each JSON
    match_json["provider"] = "europepmc"
    match_json["anns"] = match_annotations

    non_match_json["provider"] = "europepmc"
    non_match_json["anns"] = non_match_annotations

    return match_json, non_match_json


def count_lines_in_gzip(file_path):
    """Counts the lines in a gzipped file."""
    with gzip.open(file_path, "rt") as f:
        return sum(1 for _ in f)

#%%
# Main function for processing article and generating JSONs
def process_article_generate_jsons(article_data, parallel=PARALLEL_):
    article_type = article_data.get("article_type", "").lower()
    if "research" not in article_type:
        # print(article_type)
        return None, None  # Skip non-research articles

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

    if predicted_label != "metagenomics" or (predicted_label == "metagenomics" and proba<0.65):
        return None, None  # Skip NER tagging if label is "other"
    
    all_annotations = []
    print([article_type,pmcid,predicted_label, proba])
    # Loop through each NER model for the specified sections
    for model in ner_models:
        # print(model)
        for section_key, sentences in article_data.get("sections", {}).items():
            if section_key not in ["INTRO", "METHODS", "RESULTS", "DISCUSS"]:
                continue  # Skip all sections except the specified ones

            section = SECTIONS_MAP.get(section_key, "Other")
            # Pass the NER model and parallel flag to batch_annotate_sentences
            batch_annotations = batch_annotate_sentences(sentences, section, ner_model=model, parallel=parallel)
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

    return match_json, non_match_json

#%%
# Adjustments to `process_each_article` to ensure compatibility with the modified `process_article_generate_jsons`

def process_each_article(input_file, output_dir):
    # Create base output filenames based on input filename
    input_filename = os.path.basename(input_file).replace(".json.gz", "")
    output_file = os.path.join(output_dir, f"{input_filename}.json")
    no_match_file_path = os.path.join(output_dir, "no_matches", f"{input_filename}_no_match.json")

    # Count the total number of lines for the progress bar
    total_lines = count_lines_in_gzip(input_file)

    # Ensure no_matches directory exists
    os.makedirs(os.path.join(output_dir, "no_matches"), exist_ok=True)

    with gzip.open(input_file, "rt") as infile:
        # Use tqdm with the total number of lines
        for line in tqdm(infile, desc="Processing lines", unit="line", total=total_lines):
            article_data = json.loads(line)

            # Generate matched and unmatched JSONs
            match_json, non_match_json = process_article_generate_jsons(article_data)

            # Write match_json to the output file if it exists
            if match_json and match_json["anns"]:
                with open(output_file, "a", encoding="utf-8") as match_file:
                    json.dump(match_json, match_file, ensure_ascii=False)
                    match_file.write("\n")

            # Write non_match_json to the no_match file if it exists
            if non_match_json and non_match_json["anns"]:
                with open(no_match_file_path, "a", encoding="utf-8") as no_match_file_handler:
                    json.dump(non_match_json, no_match_file_handler, ensure_ascii=False)
                    no_match_file_handler.write("\n")

    print(f"Processing completed. Results saved to {output_file}")
    print(f"No match results saved to {no_match_file_path}")



import torch
from torch.nn.functional import softmax

def classify_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    article_model.eval()
    with torch.no_grad():
        outputs = article_model(**inputs)
    logits = outputs.logits
    
    # Calculate probabilities from logits
    probabilities = softmax(logits, dim=1)
    
    # Get the predicted class and its probability
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    predicted_probability = probabilities[0, predicted_class].item()
    
    return predicted_label, predicted_probability

#%%
    # Set ONNX runtime session options
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1  # Limit to a single thread
    session_options.inter_op_num_threads = 1  # Limit to a single thread

    # Load environment variables
    load_dotenv('/home/stirunag/work/github/CAPITAL/daily_pipeline/.env_paths')
    ml_model_path = os.getenv("MODEL_PATH_QUANTIZED")

    # Define paths
    article_model_path = ml_model_path + 'article_classifier/best_contrastive_binary_model_absurd-sweep-98.pth'
    article_label_encoder_path = ml_model_path + 'article_classifier/best_label_encoder_absurd-sweep-98.pkl'
    pretrained_model_name = "bioformers/bioformer-8L"

    # Define paths
    metagenomic_paths = [
        ml_model_path + f'metagenomics/metagenomic-set-{i}_quantised' for i in range(1, 6)
    ]
    article_model_path = ml_model_path + 'article_classifier/best_contrastive_binary_model_absurd-sweep-98.pth'
    article_label_encoder_path = ml_model_path + 'article_classifier/best_label_encoder_absurd-sweep-98.pkl'
    pretrained_model_name = "bioformers/bioformer-8L"

    # Function to load an ONNX NER model
    def load_ner_model(model_path, session_options):
        model = ORTModelForTokenClassification.from_pretrained(
            model_path, file_name="model_quantized.onnx", session_options=session_options
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=512, batch_size=4, truncation=True
        )
        return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="max")

    # Load all NER models
    ner_models = [load_ner_model(path, session_options) for path in metagenomic_paths]
    print("All NER models loaded successfully.")

    # Load the article classifier model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    article_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    article_model.load_state_dict(torch.load(article_model_path, map_location=torch.device('cpu')), strict=False)

    # Load the label encoder
    with open(article_label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    print("Article classifier model and label encoder loaded successfully.")

#%%
# Directly assign the paths
input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch_2024_10_28_0.json.gz"  # Replace with your actual input file path
output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/metagenomics/"  # Replace with your actual output directory path
#%%
# Check that input is a file
if not os.path.isfile(input_path):
    raise ValueError(f"Expected a file for input, but got: {input_path}")

# Check if output directory exists; if not, create it
if not os.path.isdir(output_path):
    print(f"Output directory '{output_path}' does not exist. Creating it.")
    os.makedirs(output_path, exist_ok=True)

# Ensure 'no_matches' directory exists within the output directory
no_match_dir = os.path.join(output_path, "no_matches")
os.makedirs(no_match_dir, exist_ok=True)
no_match_file_path = os.path.join(no_match_dir, "patch_no_match.json")


#%%
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import requests


def normalize_tag(trm, entity):
    # Set default URL to "#" to indicate no match if no updates are made later in the function
    ourl = "#"
    
    try:
        if entity not in ['primer', 'kit']:
            url = 'https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate?propertyValue=' + trm         
            response = requests.get(url)
            
            # Check for successful response
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
        
        elif entity == 'kit':
            trm = trm.replace(' ', '-')
            ourl = 'https://www.biocompare.com/General-Search/?search=' + trm

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
        pass

    return ourl

#%%
input_file =input_path
output_dir=output_path
#%%
input_filename = os.path.basename(input_file).replace(".json.gz", "")
output_file_OA = os.path.join(output_dir, f"{input_filename}_OA.json")
output_file_NOA = os.path.join(output_dir, f"{input_filename}_NOA.json")
no_match_file_OA = os.path.join(output_dir, "no_matches", f"{input_filename}_OA_no_match.json")
no_match_file_NOA = os.path.join(output_dir, "no_matches", f"{input_filename}_NOA_no_match.json")

total_lines = count_lines_in_gzip(input_file)

with gzip.open(input_file, "rt") as infile, \
     open(output_file_OA, "w", encoding="utf-8") as outfile_OA, \
     open(output_file_NOA, "w", encoding="utf-8") as outfile_NOA:

    for line in tqdm(infile, desc="Processing lines", unit="line", total=total_lines):
        article_data = json.loads(line)
        open_status = article_data.get("open_status", "")
        suffix = "_OA" if open_status in ["O", "OA"] else "_NOA"

        # Generate matched and unmatched JSONs
        match_json, non_match_json = process_article_generate_jsons(article_data)

        # Apply modifications if the article is restricted
        if suffix == "_NOA":
            if match_json:
                match_json = modify_restricted_json(match_json, open_status)
            if non_match_json:
                non_match_json = modify_restricted_json(non_match_json, open_status)

        # Write match_json and non_match_json to their respective files if they exist
        if match_json and match_json["anns"]:
            output_file = outfile_OA if suffix == "_OA" else outfile_NOA
            json.dump(match_json, output_file, ensure_ascii=False)
            output_file.write("\n")

        if non_match_json and non_match_json["anns"]:
            no_match_file_path = no_match_file_OA if suffix == "_OA" else no_match_file_NOA
            with open(no_match_file_path, "a", encoding="utf-8") as no_match_file:
                json.dump(non_match_json, no_match_file, ensure_ascii=False)
                no_match_file.write("\n")

print(f"Processing completed. OA results saved to {output_file_OA} and NOA results saved to {output_file_NOA}")
print(f"No match results saved to {no_match_file_OA} and {no_match_file_NOA}")
#%%

#%%
