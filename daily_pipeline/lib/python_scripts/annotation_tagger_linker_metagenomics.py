import os
import re
from collections import defaultdict
import requests
import argparse
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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



# --------------------
# 0) GLOBALS
# --------------------
xtrms = ['whole','abbr.','abbr','user','users','SI','`-end','-end','total','slide','slides','did not']
additional_blacklist = ['.4%', 'E8', '11.1%', '/X', 'X']
combined_blacklist = set(xtrms + additional_blacklist)

symbs = [")","(","≥","≤",">","<","~","/",".","-","%","=","+",":","%","°C","ºC","–","±","_","[","]","″","′","’","'","‐"]
symbs2 = ["≤","≥",">","<","~","/",".","-","=","±",":","–","_","″","′","’","'","‐"]

stpwrds = list(ENGLISH_STOP_WORDS)
stpwrds2 = set(stpwrds + [w.title() for w in stpwrds])

# Entity types for which we do NOT want Zooma fallback
ZOOMA_EXCLUDE_TYPES = {"body-site", "date", "kit", "primer"}

# --------------------
# 1) ONTOLOGY PRIORITY
# --------------------
ONTOLOGY_PRIORITY = {
    "gene": ["GP", "GO"],
    "treatment": ["CD", "EFO"],
    "state": ["DS", "EFO"],
    "ecoregion": ["ENVO"],
    "host": ["OG"],
    "engineered": ["ENVO", "EFO"],
    "place": ["ENVO", "EFO"],
    "site": ["EFO", "ENVO"],
    "body-site": ["EFO"],
    "sample-material": ["CD"],
    "ls": ["EFO"],
    "lcm": ["EFO"],
    "sequencing": ["EFO"],
    "primer": ["primer", "CD"],
    "kit": [],
    "date": [],
}

# --------------------
# 2) MAIN NORMALIZATION LOGIC
# --------------------
def normalize_tag(terms, entity_type, linker=None, use_zooma=True):
    """
    Normalize a list of terms to generate URIs based on the entity type,
    skipping Zooma fallback if the entity_type is in ZOOMA_EXCLUDE_TYPES.
    """
    result = {}

    # ----------------------------------------------------
    # Pre-filter & Preprocess Terms
    # ----------------------------------------------------
    filtered_terms = []
    for term in terms:
        if not term.strip():
            result[term] = (term, "#")
            continue

        # blacklist / stopword / digit / short
        if (
            term in combined_blacklist
            or term in stpwrds2
            or term.isdigit()
            or len(term) < 3
        ):
            result[term] = (term, "#")
            continue

        cleaned = term
        if entity_type != "primer":
            for sb in symbs + symbs2:
                cleaned = cleaned.replace(sb, "")
            cleaned = cleaned.strip()
            if not cleaned:
                result[term] = (term, "#")
                continue

        filtered_terms.append((term, cleaned))

    # Separate into sets
    kit_terms = set()
    date_terms = set()
    standard_terms = set()

    for original_term, cleaned_term in filtered_terms:
        if entity_type == "kit":
            kit_terms.add(original_term)
        elif entity_type == "date":
            date_terms.add(original_term)
        else:
            standard_terms.add(original_term)

    # ----------------------------------------------------
    # 1) Handle Standard Terms
    # ----------------------------------------------------
    if standard_terms:
        std_mapped = handle_priority_based_entities(
            terms=list(standard_terms),
            entity_type=entity_type,
            linker=linker
        )
        result.update(std_mapped)

    # ----------------------------------------------------
    # 2) Handle 'kit' Terms
    # ----------------------------------------------------
    if kit_terms:
        for term in kit_terms:
            result[term] = (term, generate_kit_url(term))

    # ----------------------------------------------------
    # 3) Handle 'date' Terms
    # ----------------------------------------------------
    if date_terms:
        month_to_code = {
            "January": "80", "February": "81", "March": "82", "April": "83",
            "May": "84", "June": "85", "July": "86", "August": "87",
            "September": "88", "October": "89", "November": "91", "December": "92"
        }
        month_pattern = r"\b(" + "|".join(month_to_code.keys()) + r")\b"
        for term in date_terms:
            match = re.search(month_pattern, term, re.IGNORECASE)
            if match:
                month = match.group(1).capitalize()
                uri = f"http://purl.obolibrary.org/obo/NCIT_C1061{month_to_code[month]}"
                result[term] = (term, uri)
            else:
                result[term] = (term, "#")

    # ----------------------------------------------------
    # 4) Fallback to Zooma (optional)
    # ----------------------------------------------------
    # Only run Zooma if use_zooma==True and the entity_type is allowed
    if use_zooma and entity_type not in ZOOMA_EXCLUDE_TYPES:
        # Filter out terms that we intentionally mapped to '#' because they are
        # blacklisted, short, digits, or in stopwords
        unmapped = []
        for t, (_, uri) in result.items():
            if uri == "#":
                # Check if it's eligible for Zooma
                if (t not in combined_blacklist
                        and not t.isdigit()
                        and len(t) >= 3
                        and t not in stpwrds2
                        and t.title() not in stpwrds2):
                    unmapped.append(t)

        if unmapped:
            zooma_uris = get_batch_mappings_from_zooma(unmapped)
            for t in unmapped:
                if zooma_uris.get(t, "#") != "#":
                    result[t] = (t, zooma_uris[t])

    # ----------------------------------------------------
    # 5) Ensure coverage
    # ----------------------------------------------------
    for t in terms:
        if t not in result:
            result[t] = (t, "#")

    return result


# ---------------------------
# 3) PRIORITY-BASED MAPPING
# ---------------------------
def handle_priority_based_entities(terms, entity_type, linker):
    results = {}
    ontologies_to_try = ONTOLOGY_PRIORITY.get(entity_type, [])

    if not ontologies_to_try or not linker:
        return {t: (t, "#") for t in terms}

    mapped_by_ontology = {}
    for onto in ontologies_to_try:
        mapped_by_ontology[onto] = linker.map_terms_reverse(terms, entity_type=onto)

    for t in terms:
        final_uri = "#"
        for onto in ontologies_to_try:
            mapped_dict = mapped_by_ontology[onto]
            if t in mapped_dict:
                code, label = mapped_dict[t]
                if code != "#":
                    final_uri = linker.map_to_url(onto, code)
                    if final_uri != "#":
                        break
        results[t] = (t, final_uri)
    return results


# ---------------------------
# 4) KIT HELPER
# ---------------------------
def generate_kit_url(term):
    base_url = "https://www.biocompare.com/General-Search/?search="
    return f"{base_url}{term.replace(' ', '-')}"


# ---------------------------
# 5) ZOOMA LOOKUP
# ---------------------------
def fetch_zooma_mapping(term):
    url = "https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate"
    try:
        params = {'propertyValue': term}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            rjson = response.json()
            if rjson:
                result = rjson[0]
                links = result.get("_links", {}).get("olslinks", [])
                if links:
                    return term, links[0].get("semanticTag", "#")
        return term, "#"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching term '{term}': {e}")
        return term, "#"


def get_batch_mappings_from_zooma(terms, max_workers=2):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(fetch_zooma_mapping, terms)
    return {term: uri for term, uri in results}

##############################################################################################################

# def normalize_tag(terms, entity_type, linker=None):
#     """
#     Normalize a list of terms to generate URIs based on the entity type.
#     Applies special handling:
#     - Skips normalization for blacklist/stopwords.
#     - Removes specified symbols for non-primer entities.
#     - Processes only terms longer than two characters.
#     """
#     result = {}
#     stpwrds = list(ENGLISH_STOP_WORDS)
#     stpwrds2 = set(stpwrds + [word.title() for word in stpwrds])
#
#     # Filter and preprocess terms based on blacklist, stopwords, and symbols
#     filtered_terms = []
#     for term in terms:
#         # Skip normalization if term is blacklisted or a stopword
#         if term in combined_blacklist or term in stpwrds2 or term.isdigit() or len(term)<3:
#             result[term] = (term, "#")
#             continue
#
#         # For non-primer entities, remove symbols from the term
#         if entity_type != "primer":
#             for symbol in symbs + symbs2:
#                 term = term.replace(symbol, "")
#             # If removal leaves an empty term, normalize to "#"
#             if not term.strip():
#                 result[term] = (term, "#")
#                 continue
#
#         filtered_terms.append(term)
#
#     # Initialize sets for different entity types
#     zooma_terms = set()
#     kit_terms = set()
#     date_terms = set()
#     primer_terms = set()
#
#     # Classify terms based on entity type, processing only those >2 characters
#     for term in filtered_terms:
#         if len(term) > 2:  # Process only terms longer than two characters
#             if entity_type == "kit":
#                 kit_terms.add(term)
#             elif entity_type == "date":
#                 date_terms.add(term)
#             elif entity_type == "primer":
#                 primer_terms.add(term)
#             else:
#                 zooma_terms.add(term)
#
#     # Handle ZOOMA-eligible terms
#     if zooma_terms:
#         zooma_uris = get_batch_mappings_from_zooma(list(zooma_terms))
#         for term in zooma_terms:
#             result[term] = (term, zooma_uris.get(term, "#"))  # Default to # if no mapping
#
#     # Handle 'kit' entities
#     if kit_terms:
#         for term in kit_terms:
#             result[term] = (term, generate_kit_url(term))
#
#     # Handle 'date' entities
#     if date_terms:
#         month_to_code = {
#             "January": "80", "February": "81", "March": "82", "April": "83", "May": "84",
#             "June": "85", "July": "86", "August": "87", "September": "88", "October": "89",
#             "November": "91", "December": "92"
#         }
#         month_pattern = r"\b(" + "|".join(month_to_code.keys()) + r")\b"
#
#         for term in date_terms:
#             match = re.search(month_pattern, term, re.IGNORECASE)
#             if match:
#                 month = match.group(1).capitalize()
#                 uri = f"http://purl.obolibrary.org/obo/NCIT_C1061{month_to_code[month]}"
#                 result[term] = (month, uri)
#             else:
#                 result[term] = (term, "#")
#
#     # Handle 'primer' entities
#     if primer_terms and linker:
#         primer_uris = handle_primer_entities(list(primer_terms), linker)
#         for term in primer_terms:
#             result[term] = (term, primer_uris.get(term, "#"))
#
#     # Ensure all terms have an entry; default to "#"
#     for term in terms:
#         if term not in result:
#             result[term] = (term, "#")
#
#     # Final check for unmapped terms (send them to ZOOMA if necessary)
#     unmapped_terms = [term for term, (grounded_code, uri) in result.items() if uri == '#']
#     if unmapped_terms:
#         zooma_uris = get_batch_mappings_from_zooma(unmapped_terms)
#         for term in unmapped_terms:
#             if zooma_uris.get(term, '#') != '#':
#                 result[term] = (term, zooma_uris[term])
#
#     return result
#
# def generate_kit_url(term):
#     """
#     Generate a URL for 'kit' entities.
#
#     Args:
#         term (str): The kit term to process.
#
#     Returns:
#         str: The URL for the kit entity.
#     """
#     return f"https://www.biocompare.com/General-Search/?search={term.replace(' ', '-')}"
#
# def handle_primer_entities(terms, linker):
#     """
#     Handle special cases for 'primer' entities.
#
#     Args:
#         terms (list): List of primer terms.
#         linker (object): Linker object for term mapping.
#
#     Returns:
#         dict: Mapping of terms to URIs.
#     """
#     primer_uris = {}
#     mapped_results = linker.map_terms_reverse(entities=terms, entity_type="primer")
#     for term in terms:
#         if "primer" in mapped_results and term in mapped_results["primer"]:
#             grounded_code, _ = mapped_results["primer"][term]
#             primer_uris[term] = linker.map_to_url(entity_group="Primer", ent_id=grounded_code)
#         else:
#             primer_uris[term] = "#"
#     return primer_uris
#
# def fetch_zooma_mapping(term):
#     """
#     Fetch ontology mapping for a single term from the ZOOMA API.
#
#     Args:
#         term (str): The term to map.
#
#     Returns:
#         tuple: A tuple containing the term and its corresponding URI.
#     """
#     url = "https://www.ebi.ac.uk/spot/zooma/v2/api/services/annotate"
#     try:
#         params = {'propertyValue': term}
#         response = requests.get(url, params=params, timeout=5)
#         if response.status_code == 200:
#             rjson = response.json()
#             if rjson:
#                 # Assuming we take the first result
#                 result = rjson[0]
#                 links = result.get("_links", {}).get("olslinks", [])
#                 if links:
#                     return term, links[0].get("semanticTag", "#")
#         return term, "#"
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching term '{term}': {e}")
#         return term, "#"
#
# def get_batch_mappings_from_zooma(terms, max_workers=3):
#     """
#     Fetch ontology mappings for a list of terms from the ZOOMA API using parallel requests.
#
#     Args:
#         terms (list): List of terms to map.
#         max_workers (int): Number of threads for parallel requests.
#
#     Returns:
#         dict: A dictionary with terms as keys and their corresponding URIs as values.
#     """
#     term_uris = {}
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = executor.map(fetch_zooma_mapping, terms)
#         term_uris = {term: uri for term, uri in results}
#     return term_uris

####################################################################################################################
def generate_tags(all_annotations, linker=None, use_map_terms=False, batch_size=2):
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
                batch_results = normalize_tag(batch_terms, entity_type, linker=linker, use_zooma=True)
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

    if ft_id and 'PPR' not in ft_id:
        return None, None  # Skip article if not a preprint ft_id

    if pmcid:
        # If pmcid doesn't start with "PMC" and is purely numeric, prepend "PMC"
        if not pmcid.startswith("PMC") and pmcid.isdigit():
            pmcid = "PMC" + pmcid

    # Retrieve and join sentences from the ABSTRACT section
    abstract_section = article_data.get("sections", {}).get("ABSTRACT", [])
    # Ensure each item in abstract_section is a string by extracting the 'text' key
    abstract_text_list = [sentence["text"] for sentence in abstract_section if "text" in sentence]
    # Join the abstract text
    abstract_text = " ".join(abstract_text_list)

    # Classify the article based on the abstract
    predicted_label, proba = classify_abstract(abstract_text)
    # if predicted_label != "metagenomics" or proba > 0.80:
    if predicted_label != "metagenomics" or (predicted_label == "metagenomics" and proba <= 0.85):
        return None, None  # Skip NER tagging if label is "
    # other"
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

            if section == "Other":
                continue

            # Pass the NER model and parallel flag to batch_annotate_sentences
            batch_annotations = batch_annotate_sentences(sentences, section, ner_model=metagenomic_model, extract_annotation_fn=extract_annotation, provider=PROVIDER)

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

    # # # Load environment variables
    # load_dotenv('/hps/software/users/literature/textmining-ml/.env_paths')
    #
    # ml_model_path = os.getenv('METAGENOMIC_MODEL_PATH_QUANTIZED')
    # # primer_dictionary_path = BASE_DICTIONARY_PATH
    # article_classifier_path = os.getenv('ARTICLE_CLASSIFIER_PATH')
    #
    # linker = EntityLinker()
    # loaded_data = linker.load_annotations(['primer'])
    #
    # if not ml_model_path:
    #     raise ValueError("Environment variable 'METAGENOMIC_MODEL_PATH_QUANTIZED' not found.")
    #
    #
    # metagenomic_paths = [
    #     ml_model_path + '/' + f'metagenomic-set-{i}_quantised' for i in range(1, 6)
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
    # parser = argparse.ArgumentParser(
    #     description='Process section-tagged XML files and output annotations in JSON format.')
    # parser.add_argument('--input', help='Input directory with XML or GZ files', required=True)
    # parser.add_argument('--output', help='Output directory for JSON files', required=True)
    # # parser.add_argument('--model_path', help='Path to the quantized model directory', required=True)
    #
    # args = parser.parse_args()
    # input_path = args.input
    # output_path = args.output
    # # model_path_quantised = args.model_path
    #
    # # Check that input is a file
    # if not os.path.isfile(input_path):
    #     raise ValueError(f"Expected a file for input, but got: {input_path}")
    #
    # # Check if output directory exists; if not, create it
    # if not os.path.isdir(output_path):
    #     print(f"Output directory '{output_path}' does not exist. Creating it.")
    #     os.makedirs(output_path, exist_ok=True)
    # #
    # # Ensure 'no_matches' directory exists within the output directory
    # no_match_dir = os.path.join(output_path, "no_matches")
    # os.makedirs(no_match_dir, exist_ok=True)
    # # no_match_file_path = os.path.join(no_match_dir, "patch_no_match.json")
    # #
    # process_each_article(
    #     input_file=input_path,
    #     output_dir=output_path,
    #     process_article_json_fn=process_article_generate_jsons,
    # )

    ######################################################################################################

    ml_model_path = '/home/stirunag/work/github/CAPITAL/model/'
    primer_dictionary_path = '/home/stirunag/work/github/CAPITAL/normalisation/dictionary/'
    article_classifier_path = ml_model_path+"article_classifier/"

    # Instantiate the EntityLinker class
    linker = EntityLinker()
    loaded_data = linker.load_annotations(['primer', 'GP', 'DS', 'OG', 'CD', 'EFO', 'ENVO', 'EM', 'GO'])

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
    input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch-2024_10_29-15.jsonl.gz"  # Replace with your actual input file path
    output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/fulltext/metagenomics/"  # Replace with your actual output directory path

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
############################################################################################################



