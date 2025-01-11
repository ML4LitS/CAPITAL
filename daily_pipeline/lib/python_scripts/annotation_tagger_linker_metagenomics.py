import os
import re
from collections import defaultdict
import requests
import argparse
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from entity_tagger import load_ner_model, load_artifacts, classify_abstract, process_each_article, batch_annotate_sentences, extract_annotation, format_output_annotations, SECTIONS_MAP
# from entity_linker import load_annotations, map_to_url, map_terms_reverse
from entity_linker import EntityLinker

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    # "sample-material": "Sample-Material",
    # "body-site": "Body-Site",
    # "host": "Host",
    # "state": "State",
    # "site": "Site",
    # "place": "Place",
    # "date": "Date",
    # "engineered": "Engineered",
    # "ecoregion": "Ecoregion",
    # "treatment": "Treatment",
    # "kit": "Kit",
    # "primer": "Primer",
    # "gene": "Gene",
    # "ls": "LS",
    # "lcm": "LCM",
    # "sequencing": "Sequencing"
}


# def map_entity_type(abbrev, ENTITY_TYPE_MAP):
#     """Map abbreviation to full form."""
#     return ENTITY_TYPE_MAP.get(abbrev, abbrev.lower())

def normalize_tag(terms, entity_type, linker=None):
    """
    Normalize a list of terms to generate URIs based on the entity type.
    Combines batch processing for ZOOMA with special handling for specific entities.

    Args:
        terms (list): List of terms to normalize.
        entity_type (str): The entity type (e.g., 'primer', 'kit', 'date').
        linker (object): Linker object for term mapping (required for primer entities).

    Returns:
        dict: A mapping of terms to (grounded_code, uri) pairs.
    """
    result = {}

    # Initialize sets for different entity types
    zooma_terms = set()
    kit_terms = set()
    date_terms = set()
    primer_terms = set()

    # Classify terms based on entity type
    for term in terms:
        if len(term)>2 or term in ['.4%', 'E8', '11.1%']:
            if entity_type == "kit":
                kit_terms.add(term)
            elif entity_type == "date":
                date_terms.add(term)
            elif entity_type == "primer":
                primer_terms.add(term)
            else:
                zooma_terms.add(term)

    # Handle ZOOMA-eligible terms
    if zooma_terms:
        zooma_uris = get_batch_mappings_from_zooma(list(zooma_terms))
        for term in zooma_terms:
            result[term] = (term, zooma_uris.get(term, "#"))  # Default to # if no mapping

    # Handle 'kit' entities
    if kit_terms:
        for term in kit_terms:
            result[term] = (term, generate_kit_url(term))

    # Handle 'date' entities
    if date_terms:
        month_to_code = {
            "January": "80", "February": "81", "March": "82", "April": "83", "May": "84",
            "June": "85", "July": "86", "August": "87", "September": "88", "October": "89",
            "November": "91", "December": "92"
        }
        month_pattern = r"\b(" + "|".join(month_to_code.keys()) + r")\b"

        for term in date_terms:
            # Search for the first occurrence of a month in the term
            match = re.search(month_pattern, term, re.IGNORECASE)
            if match:
                month = match.group(1).capitalize()  # Extract the matched month and capitalize it
                uri = f"http://purl.obolibrary.org/obo/NCIT_C1061{month_to_code[month]}"
                result[term] = (month, uri)  # Use the month as the grounded term
            else:
                result[term] = (term, "#")  # Default URI if no month is found

    # Handle 'primer' entities
    if primer_terms and linker:
        primer_uris = handle_primer_entities(list(primer_terms), linker)
        for term in primer_terms:
            result[term] = (term, primer_uris.get(term, "#"))

    # Ensure all terms are accounted for with a default value
    for term in terms:
        if term not in result:
            result[term] = (term, "#")

# Final checking of unmapped terms especially Primer and Date
    unmapped_terms = [term for term, (grounded_code, uri) in result.items() if uri == '#']

    # Send unmapped terms to ZOOMA
    if unmapped_terms:
        zooma_uris = get_batch_mappings_from_zooma(unmapped_terms)
        for term in unmapped_terms:
            if zooma_uris.get(term, '#') != '#':
                result[term] = (term, zooma_uris[term])

    return result

def generate_kit_url(term):
    """
    Generate a URL for 'kit' entities.

    Args:
        term (str): The kit term to process.

    Returns:
        str: The URL for the kit entity.
    """
    return f"https://www.biocompare.com/General-Search/?search={term.replace(' ', '-')}"

def handle_primer_entities(terms, linker):
    """
    Handle special cases for 'primer' entities.

    Args:
        terms (list): List of primer terms.
        linker (object): Linker object for term mapping.

    Returns:
        dict: Mapping of terms to URIs.
    """
    primer_uris = {}
    mapped_results = linker.map_terms_reverse(entities=terms, entity_type="primer")
    for term in terms:
        if "primer" in mapped_results and term in mapped_results["primer"]:
            grounded_code, _ = mapped_results["primer"][term]
            primer_uris[term] = linker.map_to_url(entity_group="Primer", ent_id=grounded_code)
        else:
            primer_uris[term] = "#"
    return primer_uris

def fetch_zooma_mapping(term):
    """
    Fetch ontology mapping for a single term from the ZOOMA API.

    Args:
        term (str): The term to map.

    Returns:
        tuple: A tuple containing the term and its corresponding URI.
    """
    url = "https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate"
    try:
        params = {'propertyValue': term}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            rjson = response.json()
            if rjson:
                # Assuming we take the first result
                result = rjson[0]
                links = result.get("_links", {}).get("olslinks", [])
                if links:
                    return term, links[0].get("semanticTag", "#")
        return term, "#"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching term '{term}': {e}")
        return term, "#"

def get_batch_mappings_from_zooma(terms, max_workers=3):
    """
    Fetch ontology mappings for a list of terms from the ZOOMA API using parallel requests.

    Args:
        terms (list): List of terms to map.
        max_workers (int): Number of threads for parallel requests.

    Returns:
        dict: A dictionary with terms as keys and their corresponding URIs as values.
    """
    term_uris = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(fetch_zooma_mapping, terms)
        term_uris = {term: uri for term, uri in results}
    return term_uris


def generate_tags(all_annotations, linker=None, use_map_terms=False, batch_size=10):
    """
    Generate tags for each annotation in all_annotations. Handles provider-specific requirements.

    Args:
        all_annotations (list): List of annotations to process.
        linker (object): Linker object for term mapping (required for map_terms_reverse).
        use_map_terms (bool): Whether to use map_terms_reverse or normalize_tag for all entities.
        batch_size (int): Maximum number of terms to process in a single batch.

    Returns:
        list: Processed annotations with tags (name and uri).
    """
    output_annotations = []
    mapped_results = {}

    # Group annotations by type for batch processing
    entities_by_type = defaultdict(list)
    for annotation in all_annotations:
        entities_by_type[annotation['type']].append(annotation['exact'])

    # Batch processing for Provider 2 (metagenomics)
    if not use_map_terms:
        for entity_type, terms in entities_by_type.items():
            entity_mapped_results = {}

            # Process terms in batches
            for i in range(0, len(terms), batch_size):
                batch_terms = terms[i:i + batch_size]
                batch_results = normalize_tag(batch_terms, entity_type, linker=linker)
                entity_mapped_results.update({
                    term: (grounded_code, uri) for term, (grounded_code, uri) in batch_results.items()
                })

            mapped_results[entity_type] = entity_mapped_results

    # Assign tags to each annotation
    for annotation in all_annotations:
        entity_type = annotation['type']
        term = annotation['exact']
        uri = "#"
        grounded_term = term  # Default to the term itself

        # Retrieve mapped results
        entity_mapped_results = mapped_results.get(entity_type, {})
        if term in entity_mapped_results:
            grounded_code, uri = entity_mapped_results[term]
            grounded_term = grounded_code

        # Append processed annotation
        output_annotations.append({
            "type": entity_type,
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
    if predicted_label != "metagenomics" or proba > 0.85:
        return None, None  # Skip NER tagging if label is "other"
    else:
        print([article_type, open_status, pmcid, predicted_label, proba])

    all_annotations = []
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
    all_linked_annotations = generate_tags(all_annotations, linker=linker)
    # Format matched and unmatched JSON structures
    match_json, non_match_json = format_output_annotations(all_linked_annotations, pmcid=pmcid, ft_id=ft_id, PROVIDER=PROVIDER)

    # Return None if both JSONs are empty or have empty 'anns' lists
    if not match_json["anns"] and not non_match_json["anns"]:
        return None, None
    #
    # print([abstract_text, article_type, open_status, pmcid, predicted_label, proba])
    return match_json, non_match_json


# Main entry point with updated argument parsing
if __name__ == '__main__':
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1  # Limit to a single thread
    session_options.inter_op_num_threads = 1  # Limit to a single thread

    # Load environment variables
    load_dotenv('/hps/software/users/literature/textmining-ml/.env_paths')

    ######################################################################################################

    # ml_model_path = '/home/stirunag/work/github/CAPITAL/model/'
    # primer_dictionary_path = '/home/stirunag/work/github/CAPITAL/normalisation/dictionary/'
    # article_classifier_path = ml_model_path+"article_classifier/"
    #
    # # Instantiate the EntityLinker class
    # linker = EntityLinker()
    # loaded_data = linker.load_annotations(['primer'])
    #
    # if not ml_model_path:
    #     raise ValueError("Environment variable 'MODEL_PATH_QUANTIZED' not found.")
    #
    #
    # metagenomic_paths = [
    #     ml_model_path + f'metagenomics/metagenomic-set-{i}_quantised' for i in range(1, 6)
    # ]
    #
    # # Load all NER models
    # try:
    #     ner_models = [load_ner_model(path, session_options) for path in metagenomic_paths]
    #     print("All NER models loaded successfully.")
    # except Exception as e:
    #     raise RuntimeError(f"Error loading NER models: {str(e)}")
    #
    # # Load all NER models
    # try:
    #     load_artifacts(article_classifier_path)
    #     print("All article classifier models loaded successfully.")
    # except Exception as e:
    #     raise RuntimeError(f"Error loading article classifier models: {str(e)}")
    #
    #
    #
    # # Define paths
    # input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch_2024_10_28_0.json_old.gz"  # Replace with your actual input file path
    # output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/fulltext/metagenomics/"  # Replace with your actual output directory path
    #
    # # Check paths
    # if not os.path.isfile(input_path):
    #     raise FileNotFoundError(f"Input file not found: {input_path}")
    # if not os.path.isdir(output_path):
    #     os.makedirs(output_path, exist_ok=True)
    #
    # # Process articles
    # process_each_article(
    #     input_file=input_path,
    #     output_dir=output_path,
    #     process_article_json_fn=process_article_generate_jsons,
    # )
############################################################################################################
    ml_model_path = os.getenv('METAGENOMIC_MODEL_PATH_QUANTIZED')
    # primer_dictionary_path = BASE_DICTIONARY_PATH
    article_classifier_path = os.getenv('ARTICLE_CLASSIFIER_PATH')

    linker = EntityLinker()
    loaded_data = linker.load_annotations(['primer'])

    if not ml_model_path:
        raise ValueError("Environment variable 'METAGENOMIC_MODEL_PATH_QUANTIZED' not found.")


    metagenomic_paths = [
        ml_model_path + '/' + f'metagenomic-set-{i}_quantised' for i in range(1, 6)
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


    parser = argparse.ArgumentParser(
        description='Process section-tagged XML files and output annotations in JSON format.')
    parser.add_argument('--input', help='Input directory with XML or GZ files', required=True)
    parser.add_argument('--output', help='Output directory for JSON files', required=True)
    # parser.add_argument('--model_path', help='Path to the quantized model directory', required=True)

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    # model_path_quantised = args.model_path

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
    process_each_article(
        input_file=input_path,
        output_dir=output_path,
        process_article_json_fn=process_article_generate_jsons,
    )



