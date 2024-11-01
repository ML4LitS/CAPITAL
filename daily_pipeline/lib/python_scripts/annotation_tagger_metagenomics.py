import json
import os
import sys
import gzip
from collections import defaultdict, OrderedDict, Counter
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForTokenClassification
import argparse
from tqdm import tqdm
import onnxruntime as ort
from entity_linker import map_to_url, map_terms, map_terms_reverse, get_exact_match, get_embedding_match, clean_term,$

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



PROVIDER = "europepmc"
PARALLEL_ = True


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

from itertools import islice

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
def batch_annotate_sentences(sentences, section, parallel=PARALLEL_):
    annotations = []

    if parallel:
        # Process in batches of at least 4 sentences
        for sentence_batch in batch_sentences(sentences, n=4):
            batched_text = [s["text"] for s in sentence_batch]
            ner_results = ner_quantized(batched_text)

            for i, sentence_entities in enumerate(ner_results):
                sentence_id = sentence_batch[i]["sent_id"]
                sentence_text = sentence_batch[i]["text"]
                for entity in sentence_entities:
                    annotations.append(extract_annotation(sentence_id, sentence_text, entity, section))
    else:
        # Process sentences individually
        for sentence in sentences:
            ner_results = ner_quantized([sentence["text"]])[0]
            sentence_id = sentence["sent_id"]
            sentence_text = sentence["text"]
            for entity in ner_results:
                annotations.append(extract_annotation(sentence_id, sentence_text, entity, section))

    return annotations

def generate_tags(all_annotations):
    """
    Generate tags for each annotation in all_annotations using map_terms_reverse and map_to_url.
    Each annotation will have 'name' and 'uri' fields in the 'tags' list.
    """
    output_annotations = []

    # Group entities by type for map_terms_reverse
    entities_by_type = defaultdict(set)
    for annotation in all_annotations:
        entities_by_type[annotation['type']].add(annotation['exact'])

    # Process each entity type with map_terms_reverse to get mapped terms and URLs
    mapped_results = {}
    for entity_type, entities in entities_by_type.items():
        mapped_results[entity_type] = map_terms_reverse(entities, entity_type)

    # Generate tags for each annotation
    for annotation in all_annotations:
        entity_type = annotation['type']
        term = annotation['exact']

        # Retrieve grounded code and term from mapped results
        if term in mapped_results[entity_type]:
            grounded_code, grounded_term = mapped_results[entity_type][term]
            uri = map_to_url(entity_type, grounded_code)  # Generate URI based on entity group and code

            # Add the annotation with tags
            output_annotations.append({
                "type": map_entity_type(entity_type),
                "position": annotation["position"],
                "prefix": annotation["prefix"],
                "exact": term,
                "section": annotation["section"],
                "postfix": annotation["postfix"],
                "tags": [
                    {
                        "name": grounded_term,
                        "uri": uri
                    }
                ]
            })
        else:
            # In case there’s no mapping found, skip or add with no URI
            output_annotations.append({
                "type": map_entity_type(entity_type),
                "position": annotation["position"],
                "prefix": annotation["prefix"],
                "exact": term,
                "section": annotation["section"],
                "postfix": annotation["postfix"],
                "tags": [
                    {
                        "name": "#",
                        "uri": "#"
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
        if annotation["tags"][0]["name"] == "#" and annotation["tags"][0]["uri"].endswith("#"):
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


# Main function for processing article and generating JSONs
def process_article_generate_jsons(article_data, parallel=PARALLEL_):
    pmcid = article_data.get("article_ids", {}).get("pmcid")
    ft_id = article_data.get("article_ids", {}).get("archive") or article_data.get("article_ids", {}).get("manuscript")

    if not pmcid and not ft_id:
        return None, None  # Skip article if no pmcid or ft_id

    all_annotations = []
    for section_key, sentences in article_data.get("sections", {}).items():
        if section_key == "REF":
            continue  # Skip processing this section
        section = SECTIONS_MAP.get(section_key, "Other")

        # Pass the parallel flag to batch_annotate_sentences
        batch_annotations = batch_annotate_sentences(sentences, section, parallel=parallel)
        if not batch_annotations:
            continue

        all_annotations.extend(batch_annotations)

    # Generate tags for annotations- this includes grounded terms and grounded codes.
    all_linked_annotations = generate_tags(all_annotations)
    # Format matched and unmatched JSON structures
    match_json, non_match_json = format_output_annotations(all_linked_annotations, pmcid=pmcid, ft_id=ft_id)

    # Return None if both JSONs are empty or have empty 'anns' lists
    if not match_json["anns"] and not non_match_json["anns"]:
        return None, None

    return match_json, non_match_json


def count_lines_in_gzip(file_path):
    """Counts the lines in a gzipped file."""
    with gzip.open(file_path, "rt") as f:
        return sum(1 for _ in f)

def process_each_article(input_file, output_dir):
    # Create base output filenames based on input filename
    input_filename = os.path.basename(input_file).replace(".json.gz", "")
    output_file_OA = os.path.join(output_dir, f"{input_filename}_OA.json")
    output_file_NOA = os.path.join(output_dir, f"{input_filename}_NOA.json")

    # Similarly, create no match filenames
    no_match_file_OA = os.path.join(output_dir, "no_matches", f"{input_filename}_OA_no_match.json")
    no_match_file_NOA = os.path.join(output_dir, "no_matches", f"{input_filename}_NOA_no_match.json")

    # Count the total number of lines for the progress bar
    total_lines = count_lines_in_gzip(input_file)

    with gzip.open(input_file, "rt") as infile, \
         open(output_file_OA, "w", encoding="utf-8") as outfile_OA, \
         open(output_file_NOA, "w", encoding="utf-8") as outfile_NOA:

        # Use tqdm with the total number of lines
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



# Main entry point with updated argument parsing
if __name__ == '__main__':

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1  # Limit to a single thread
    session_options.inter_op_num_threads = 1  # Limit to a single thread

    # # Directly assign the paths
    # input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch_2024_10_28_0.json.gz"  # Replace with your actual input file path
    # output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results"  # Replace with your actual output directory path
    # model_path_quantised = "/home/stirunag/work/github/CAPITAL/model"  # Replace with your actual model directory path
    #
    # # Check that input is a file
    # if not os.path.isfile(input_path):
    #     raise ValueError(f"Expected a file for input, but got: {input_path}")
    #
    # # Check if output directory exists; if not, create it
    # if not os.path.isdir(output_path):
    #     print(f"Output directory '{output_path}' does not exist. Creating it.")
    #     os.makedirs(output_path, exist_ok=True)
    #
    # # Ensure 'no_matches' directory exists within the output directory
    # no_match_dir = os.path.join(output_path, "no_matches")
    # os.makedirs(no_match_dir, exist_ok=True)
    # no_match_file_path = os.path.join(no_match_dir, "patch_no_match.json")
    #
    # # Initialize NER model using the provided model path
    # print("Loading NER model and tokenizer from " + model_path_quantised)
    # model_quantized = ORTModelForTokenClassification.from_pretrained(
    #     model_path_quantised, file_name="model_quantized.onnx", session_options=session_options)
    # tokenizer_quantized = AutoTokenizer.from_pretrained(
    #     model_path_quantised,
    #     model_max_length=512,
    #     batch_size=4,
    #     truncation=True
    # )
    # ner_quantized = pipeline(
    #     "token-classification",
    #     model=model_quantized,
    #     tokenizer=tokenizer_quantized,
    #     aggregation_strategy="max"
    # )
    # print("NER model and tokenizer loaded successfully.")
    #
    # # Now call process_each_article with input and output directories
    # process_each_article(input_path, output_path)





    parser = argparse.ArgumentParser(
        description='Process section-tagged XML files and output annotations in JSON format.')
    parser.add_argument('--input', help='Input directory with XML or GZ files', required=True)
    parser.add_argument('--output', help='Output directory for JSON files', required=True)
    parser.add_argument('--model_path', help='Path to the quantized model directory', required=True)
    args = parser.parse_args()

    # Check that input is a file and output is a directory
    if not os.path.isfile(args.input):
        raise ValueError(f"Expected a file for --input, but got: {args.input}")
    if not os.path.isdir(args.output):
        raise ValueError(f"Expected a directory for --output, but got: {args.output}")


    # Ensure 'no_matches' directory exists
    no_match_dir = os.path.join(args.output, "no_matches")
    os.makedirs(no_match_dir, exist_ok=True)
    no_match_file_path = os.path.join(no_match_dir, "patch_no_match.json")

    # Initialize NER model using the provided model path

    model_path_quantised = args.model_path

    print("Loading NER model and tokenizer loaded from "+ model_path_quantised)
    model_quantized = ORTModelForTokenClassification.from_pretrained(
        model_path_quantised, file_name="model_quantized.onnx", session_options=session_options)
    tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised,
                                    model_max_length=512, batch_size=4,truncation=True)
    ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized,
                             aggregation_strategy="max")
    print("NER model and tokenizer loaded successfully.")



    # Now call process_each_article with input and output directories
    process_each_article(args.input, args.output)

