import gzip
import os
from collections import defaultdict, OrderedDict, Counter
from itertools import islice
from typing import Callable
from tqdm import tqdm
import json
import pickle
import torch
from transformers import AutoTokenizer, pipeline, AutoModel
from optimum.onnxruntime import ORTModelForTokenClassification
import torch.nn.functional as F

BATCH_SIZE_=2
PARALLEL_=False

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

# Function to load an ONNX NER model
def load_ner_model(path, session_options):
    """
    Load a quantized NER model with ONNXRuntime.
    """
    print(f"Loading NER model from {path}")
    model = ORTModelForTokenClassification.from_pretrained(
        path, file_name="model_quantized.onnx", session_options=session_options)
    tokenizer = AutoTokenizer.from_pretrained(
        path, model_max_length=512, batch_size=1, truncation=True)
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="max"
    )
    print(f"NER model from {path} loaded successfully.")
    return ner_pipeline

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

def batch_annotate_sentences(
    sentences,
    section,
    ner_model,
    extract_annotation_fn: Callable,
    batch_size=BATCH_SIZE_,
    parallel=PARALLEL_
):
    """
    Annotate sentences in batches

    Args:
        sentences (list): List of sentences to process.
        section (str): Section name to include in annotations.
        ner_model (Callable): NER model or pipeline for extracting entities.
        extract_annotation_fn (Callable): Function to extract annotations.
        batch_size (int): Number of sentences per batch (default: 4).
        parallel (bool): Whether to process sentences in parallel (default: True).

    Returns:
        list: Annotated sentences.
    """
    annotations = []

    def batch_sentences(iterable, n):
        """Helper function to batch sentences into chunks of size n."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    if parallel:
        # Process in batches
        for sentence_batch in batch_sentences(sentences, n=batch_size):
            batched_text = [s["text"] for s in sentence_batch]
            ner_results = ner_model(batched_text)

            for i, sentence_entities in enumerate(ner_results):
                sentence_id = sentence_batch[i]["sent_id"]
                sentence_text = sentence_batch[i]["text"]
                for entity in sentence_entities:
                    annotations.append(extract_annotation_fn(sentence_id, sentence_text, entity, section))
    else:
        # Process sentences individually
        for sentence in sentences:
            ner_results = ner_model([sentence["text"]])[0]
            sentence_id = sentence["sent_id"]
            sentence_text = sentence["text"]
            for entity in ner_results:
                annotations.append(extract_annotation_fn(sentence_id, sentence_text, entity, section))

    return annotations


def format_output_annotations(all_linked_annotations_, pmcid, ft_id, PROVIDER):
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
        if annotation["tags"][0]["name"] == "#" or annotation["tags"][0]["uri"].endswith("#"):
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
    match_json["provider"] = PROVIDER
    match_json["anns"] = match_annotations

    non_match_json["provider"] = PROVIDER
    non_match_json["anns"] = non_match_annotations

    return match_json, non_match_json


def modify_restricted_json(json_data, open_status):
    """
    Modifies the JSON structure for restricted access:
    - Removes 'prefix' and 'postfix' fields.
    - Adds 'frequency' field to count occurrences of each unique 'exact' term.
    """
    if open_status in ["OA", "O"]:
        return json_data  # No modification if open access

    # Count frequency of each 'exact' term
    frequency_counter = defaultdict(int)
    for annotation in json_data["anns"]:
        frequency_counter[annotation["exact"]] += 1

    # Modify each annotation
    restricted_annotations = []
    for annotation in json_data["anns"]:
        # Remove 'prefix' and 'postfix' fields, add 'frequency'
        restricted_annotation = {
            "exact": annotation["exact"],
            "tags": annotation["tags"],
            "type": annotation["type"],
            "section": annotation["section"],
            "provider": json_data["provider"],
            "frequency": frequency_counter[annotation["exact"]]
        }
        restricted_annotations.append(restricted_annotation)

    # Build the modified JSON structure
    modified_json = OrderedDict()
    modified_json["provider"] = json_data["provider"]
    modified_json["anns"] = restricted_annotations

    # Preserve 'pmcid' or 'ft_id' in the modified JSON
    if "pmcid" in json_data:
        modified_json["pmcid"] = json_data["pmcid"]
    elif "ft_id" in json_data:
        modified_json["ft_id"] = json_data["ft_id"]

    return modified_json

def count_lines_in_gzip(file_path):
    """Counts the lines in a gzipped file."""
    with gzip.open(file_path, "rt") as f:
        return sum(1 for _ in f)

def process_each_article(input_file, output_dir, process_article_json_fn):
    """
    Process each article in an input file and save results to output files.

    Args:
        input_file (str): Path to the input file (JSON or JSON.GZ).
        output_dir (str): Directory to save output files.
        process_article_json_fn (Callable): Function to process individual articles.

    Returns:
        None
    """
    input_filename = os.path.basename(input_file).replace(".json.gz", "")
    output_file = os.path.join(output_dir, f"{input_filename}.json")
    no_match_file_path = os.path.join(output_dir, "no_matches", f"{input_filename}_no_match.json")

    # Ensure the no_matches directory exists
    os.makedirs(os.path.join(output_dir, "no_matches"), exist_ok=True)

    # Count total lines for progress bar
    total_lines = count_lines_in_gzip(input_file)

    with gzip.open(input_file, "rt") as infile:
        for line in tqdm(infile, desc="Processing lines", unit="line", total=total_lines):
            article_data = json.loads(line)

            # Process the article using the provided function
            match_json, non_match_json = process_article_json_fn(article_data)

            # Save matched JSON to the output file
            if match_json and match_json["anns"]:
                with open(output_file, "a", encoding="utf-8") as match_file:
                    json.dump(match_json, match_file, ensure_ascii=False)
                    match_file.write("\n")

            # Save unmatched JSON to the no_matches file
            if non_match_json and non_match_json["anns"]:
                with open(no_match_file_path, "a", encoding="utf-8") as no_match_file:
                    json.dump(non_match_json, no_match_file, ensure_ascii=False)
                    no_match_file.write("\n")

    print(f"Processing completed. Results saved to {output_file}")
    print(f"No match results saved to {no_match_file_path}")


# def process_each_article_suffix(input_file, output_dir):
#     # Create base output filenames based on input filename
#     input_filename = os.path.basename(input_file).replace(".json.gz", "")
#     output_file_OA = os.path.join(output_dir, f"{input_filename}_OA.json")
#     output_file_NOA = os.path.join(output_dir, f"{input_filename}_NOA.json")
#
#     # Similarly, create no match filenames
#     no_match_file_OA = os.path.join(output_dir, "no_matches", f"{input_filename}_OA_no_match.json")
#     no_match_file_NOA = os.path.join(output_dir, "no_matches", f"{input_filename}_NOA_no_match.json")
#
#     # Count the total number of lines for the progress bar
#     total_lines = count_lines_in_gzip(input_file)
#
#     with gzip.open(input_file, "rt") as infile, \
#          open(output_file_OA, "w", encoding="utf-8") as outfile_OA, \
#          open(output_file_NOA, "w", encoding="utf-8") as outfile_NOA:
#
#         # Use tqdm with the total number of lines
#         for line in tqdm(infile, desc="Processing lines", unit="line", total=total_lines):
#             article_data = json.loads(line)
#             open_status = article_data.get("open_status", "")
#             suffix = "_OA" if open_status in ["O", "OA"] else "_NOA"
#
#             # Generate matched and unmatched JSONs
#             match_json, non_match_json = process_article_generate_jsons(article_data)
#
#             # Apply modifications if the article is restricted
#             if suffix == "_NOA":
#                 if match_json:
#                     match_json = modify_restricted_json(match_json, open_status)
#                 if non_match_json:
#                     non_match_json = modify_restricted_json(non_match_json, open_status)
#
#             # Write match_json and non_match_json to their respective files if they exist
#             if match_json and match_json["anns"]:
#                 output_file = outfile_OA if suffix == "_OA" else outfile_NOA
#                 json.dump(match_json, output_file, ensure_ascii=False)
#                 output_file.write("\n")
#
#             if non_match_json and non_match_json["anns"]:
#                 no_match_file_path = no_match_file_OA if suffix == "_OA" else no_match_file_NOA
#                 with open(no_match_file_path, "a", encoding="utf-8") as no_match_file:
#                     json.dump(non_match_json, no_match_file, ensure_ascii=False)
#                     no_match_file.write("\n")
#
#     print(f"Processing completed. OA results saved to {output_file_OA} and NOA results saved to {output_file_NOA}")
#     print(f"No match results saved to {no_match_file_OA} and {no_match_file_NOA}")




