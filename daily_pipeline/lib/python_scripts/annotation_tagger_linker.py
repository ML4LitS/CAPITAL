import os
import sys
from collections import defaultdict, OrderedDict, Counter
import onnxruntime as ort
from entity_linker import EntityLinker
from entity_tagger import process_each_article, load_ner_model, batch_annotate_sentences, format_output_annotations, SECTIONS_MAP, extract_annotation
import argparse
import warnings
import re
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Mapping from abbreviation to full form
ENTITY_TYPE_MAP_1 = {
    "EM": "methods", #exp_methods
    "DS": "disease",
    "GP": "gene_protein",
    "GO": "go_term",
    "CD": "chemical",
    "OG": "organism"
}

PROVIDER = "europepmc"
TYPES = list(ENTITY_TYPE_MAP_1.values())

# Helper Functions
def map_entity_type(abbrev, ENTITY_TYPE_MAP):
    """Map abbreviation to full form."""
    return ENTITY_TYPE_MAP.get(abbrev, abbrev.lower())


def generate_tags(all_annotations, entity_map, linker=None):
    """
    Generate tags for each annotation in all_annotations. Handles provider-specific requirements.

    Args:
        all_annotations (list): List of annotations to process.
        entity_map (dict): Mapping of entity abbreviations to full forms.
        linker (object): Linker object for term mapping (required for map_terms_reverse).

    Returns:
        list: Processed annotations with tags (name and uri).
    """
    output_annotations = []
    mapped_results = {}

    # Group annotations by type for batch processing
    entities_by_type = defaultdict(list)
    for annotation in all_annotations:
        entities_by_type[annotation['type']].append(annotation['exact'])

    for entity_type, terms in entities_by_type.items():
        mapped_results[entity_type] = linker.map_terms_reverse(terms, entity_type)

    # Assign tags to each annotation
    for annotation in all_annotations:
        entity_type = annotation['type']
        term = annotation['exact']
        uri = "#"
        grounded_term = "#"  # Default to the term itself

        # Retrieve mapped results
        entity_mapped_results = mapped_results.get(entity_type, {})
        if term in entity_mapped_results:
            grounded_code, grounded_term = mapped_results[entity_type][term]
            if grounded_code !='#'and grounded_term !='#':
                uri = linker.map_to_url(entity_type, grounded_code)  # Generate URI based on entity group and code


        # Append processed annotation
        output_annotations.append({
            "type": map_entity_type(entity_type, entity_map),
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

    return output_annotations

# Main function for processing article and generating JSONs
def process_article_generate_jsons(article_data):
    pmcid = article_data.get("article_ids", {}).get("pmcid")
    ft_id = article_data.get("article_ids", {}).get("archive") or article_data.get("article_ids", {}).get("manuscript")

    if not pmcid and not ft_id:
        return None, None  # Skip article if no pmcid or ft_id

    if pmcid:
        # If pmcid starts with "PMC", case-insensitively extract only the digits
        pmcid_match = re.search(r"(?i)PMC(\d+)", pmcid)  # (?i) makes the regex case-insensitive
        if pmcid_match:
            pmcid = pmcid_match.group(1)
        # Check if pmcid is a still a valid numeric string
        if not pmcid.isdigit():
            # If not numeric, assign pmcid to ft_id and reset pmcid
            ft_id = pmcid
            pmcid = None

    # print([pmcid, ft_id])
    all_annotations = []

    for section_key, sentences in article_data.get("sections", {}).items():
        # Skip explicitly unwanted section keys (case-sensitive check)
        if section_key == "REF" or section_key == "OTHER":
            continue  # Skip processing this section

        # Map the section key, defaulting to "Other" if not found
        section = SECTIONS_MAP.get(section_key, "Other")

        # Skip if the mapped section is "Other"
        if section == "Other":
            continue

        # Pass the parallel flag to batch_annotate_sentences
        batch_annotations = batch_annotate_sentences(sentences, section, ner_model=ner_quantized, extract_annotation_fn=extract_annotation)
        if not batch_annotations:
            continue

        all_annotations.extend(batch_annotations)

    # Generate tags for annotations- this includes grounded terms and grounded codes.
    all_linked_annotations = generate_tags(all_annotations, entity_map=ENTITY_TYPE_MAP_1, linker=linker)
    # Format matched and unmatched JSON structures
    match_json, non_match_json = format_output_annotations(all_linked_annotations, pmcid=pmcid, ft_id=ft_id, PROVIDER=PROVIDER, TYPES=TYPES)

    # Return None if both JSONs are empty or have empty 'anns' lists
    if not match_json["anns"] and not non_match_json["anns"]:
        return None, None

    return match_json, non_match_json

# Main entry point with updated argument parsing
if __name__ == '__main__':
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1  # Limit to a single thread
    session_options.inter_op_num_threads = 1  # Limit to a single thread

    parser = argparse.ArgumentParser(
        description='Process section-tagged XML files and output annotations in JSON format.')
    parser.add_argument('--input', help='Input directory with XML or GZ files', required=True)
    parser.add_argument('--output', help='Output directory for JSON files', required=True)
    parser.add_argument('--model_path', help='Path to the quantized model directory', required=True)

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    model_path_quantised = args.model_path

    # Check that input is a file
    if not os.path.isfile(input_path):
        raise ValueError(f"Expected a file for input, but got: {input_path}")

    # Check if output directory exists; if not, create it
    if not os.path.isdir(output_path):
        print(f"Output directory '{output_path}' does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)
    #
    # Ensure 'no_matches' directory exists within the output directory
    no_match_dir = os.path.join(output_path, "no_matches")
    os.makedirs(no_match_dir, exist_ok=True)
    # no_match_file_path = os.path.join(no_match_dir, "patch_no_match.json")
    #
    # Load PROVIDER_1 (europepmc) NER model
    print("Loading NER model and tokenizer from " + model_path_quantised)
    ner_quantized = load_ner_model(model_path_quantised, session_options)

    # Load EntityLinker with all required annotations, including primer
    print("Loading entity linker.")
    linker = EntityLinker()
    linker.load_annotations(['EM', 'DS', 'GP', 'GO', 'CD', 'OG'])

    process_each_article(
        input_file=input_path,
        output_dir=output_path,
        process_article_json_fn=process_article_generate_jsons,
    )

#################################################################################################################################
    # Directly assign the paths
    # session_options = ort.SessionOptions()
    # session_options.intra_op_num_threads = 1  # Limit to a single thread
    # session_options.inter_op_num_threads = 1  # Limit to a single thread
    #
    # input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch-2025_01_14-24.jsonl.gz" # Replace with your actual input file path
    # output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/fulltext/europepmc/"  # Replace with your actual output directory path
    # model_path_quantised = "/home/stirunag/work/github/CAPITAL/model/europepmc"  # Replace with your actual model directory path
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
    # # Load PROVIDER_1 (europepmc) NER model
    # print("Loading NER model and tokenizer from " + model_path_quantised)
    # ner_quantized = load_ner_model(model_path_quantised, session_options)
    #
    # # Load EntityLinker with all required annotations, including primer
    # print("Loading entity linker.")
    # linker = EntityLinker()
    # linker.load_annotations(['EM', 'DS', 'GP', 'GO', 'CD', 'OG'])
    #
    # process_each_article(
    #     input_file=input_path,
    #     output_dir=output_path,
    #     process_article_json_fn=process_article_generate_jsons,
    # )

################################################################################################################


