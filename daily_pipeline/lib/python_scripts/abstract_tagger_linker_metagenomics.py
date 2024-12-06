import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import spacy
from entity_linker import EntityLinker
from entity_tagger import load_ner_model, classify_abstract,  load_artifacts, batch_annotate_sentences, extract_annotation
import gzip
import re

from collections import defaultdict
from tqdm import tqdm
import onnxruntime as ort



PROVIDER = "metagenomics"

ENTITY_TYPE_MAP_2 = {
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

# Function to load an ONNX NER model
# def load_ner_model(path, session_options):
#     """
#     Load a quantized NER model with ONNXRuntime.
#     """
#     print(f"Loading NER model from {path}")
#     model = ORTModelForTokenClassification.from_pretrained(
#         path, file_name="model_quantized.onnx", session_options=session_options)
#     tokenizer = AutoTokenizer.from_pretrained(
#         path, model_max_length=512, batch_size=1, truncation=True)
#     ner_pipeline = pipeline(
#         "token-classification",
#         model=model,
#         tokenizer=tokenizer,
#         aggregation_strategy="max"
#     )
#     print(f"NER model from {path} loaded successfully.")
#     return ner_pipeline



# Helper Functions
def map_entity_type(abbrev, ENTITY_TYPE_MAP):
    """Map abbreviation to full form."""
    return ENTITY_TYPE_MAP.get(abbrev, abbrev.lower())

# Initialize spaCy model
nlp = spacy.load("en_core_sci_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
nlp.add_pipe("sentencizer")


def read_file_blocks(file_path):
    """Extract <article> blocks from a file (plain text or gzip)."""
    open_func = gzip.open if file_path.endswith(".gz") else open
    try:
        with open_func(file_path, "rt", encoding="utf8") as file:
            content = file.read()
        return re.findall(r"<article.*?>.*?</article>", content, re.DOTALL)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []


def extract_article_data(article_xml):
    """Parse XML of an <article> to extract relevant data."""
    try:
        soup = BeautifulSoup(article_xml, "lxml")

        ext_id = soup.find("ext_id").text if soup.find("ext_id") else None
        source = soup.find("source").text if soup.find("source") else None
        title = soup.find("title").text if soup.find("title") else ""
        abstract = soup.find("p").text if soup.find("p") else ""

        # Split abstract into sentences
        doc = nlp(abstract)
        sentences = [{"sent_id": i + 1, "text": sent.text.strip()} for i, sent in enumerate(doc.sents)]

        return {
            "ext_id": ext_id,
            "source": source,
            "sections": {
                "Title": [{"sent_id": 1, "text": title}] if title else [],
                "Abstract": sentences,
            },
        }
    except Exception as e:
        print(f"Error parsing article XML: {e}")
        return None

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
        if len(term)>1:
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
        response = requests.get(url, params=params, timeout=10)
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

def get_batch_mappings_from_zooma(terms, max_workers=5):
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


def generate_tags(all_annotations, entity_map, linker=None, use_map_terms=False):
    """
    Generate tags for each annotation in all_annotations. Handles provider-specific requirements.

    Args:
        all_annotations (list): List of annotations to process.
        entity_map (dict): Mapping of entity abbreviations to full forms.
        linker (object): Linker object for term mapping (required for map_terms_reverse).
        use_map_terms (bool): Whether to use map_terms_reverse or normalize_tag for all entities.
        provider (str): The provider name, used to customize behavior (e.g., "europepmc", "metagenomics").

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
            batch_results = normalize_tag(terms, entity_type, linker=linker)
            mapped_results[entity_type] = {
                term: (grounded_code, uri) for term, (grounded_code, uri) in batch_results.items()
            }

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


def format_output_annotations(all_linked_annotations, ext_id, source, provider):
    """
    Formats output annotations into matched and unmatched JSON structures.

    Args:
        all_linked_annotations (list): List of annotations with tags.
        ext_id (str): External ID of the article.
        source (str): Source of the article (e.g., "MED").
        provider (str): The provider name (e.g., "europepmc" or "metagenomics").

    Returns:
        tuple: Matched JSON and unmatched JSON structures.
    """
    match_annotations = []
    non_match_annotations = []

    # Separate annotations based on tags
    for annotation in all_linked_annotations:
        if annotation["tags"][0]["name"] == "#" or annotation["tags"][0]["uri"].endswith("#"):
            non_match_annotations.append(annotation)
        else:
            match_annotations.append(annotation)

    # Construct final JSON outputs
    match_json = {
        "src": source,
        "ext_id": ext_id,
        "provider": provider,
        "anns": match_annotations,
    }

    non_match_json = {
        "src": source,
        "ext_id": ext_id,
        "provider": provider,
        "anns": non_match_annotations,
    }

    return match_json, non_match_json



def process_and_save_articles(
    files_list,
    extract_annotation_fn,
    classify_abstract_fn,
    ner_models_metagenomics,
    output_filename,
    output_dir,
):
    """
    Process articles, saving results immediately.

    Args:
        files_list (list): List of raw article XML strings.
        ner_model: Named entity recognition model for PROVIDER_1.
        extract_annotation_fn (callable): Function to extract annotations.
        classify_abstract_fn (callable): Function to classify abstracts.
        ner_models_metagenomics (list): List of NER models for PROVIDER_2.
        output_dir (str): Directory to save results.

    Returns:
        None
    """
    provider2_dir = os.path.join(output_dir, PROVIDER)
    os.makedirs(provider2_dir, exist_ok=True)
    os.makedirs(os.path.join(provider2_dir, "no_matches"), exist_ok=True)

    for article_xml in tqdm(files_list, desc="Processing Articles"):
        article_data = extract_article_data(article_xml)
        if not article_data or "sections" not in article_data:
            print(f"Skipping malformed article: {article_xml[:100]}...")
            continue

        ext_id = article_data["ext_id"]

        ### Process PROVIDER_2 ###
        abstract_text = " ".join([sentence["text"] for sentence in article_data["sections"].get("Abstract", [])])
        predicted_label, proba = classify_abstract_fn(abstract_text)

        if predicted_label == "metagenomics" and proba>0.85:
            all_annotations_p2 = []
            for section_name, sentences in article_data["sections"].items():
                for metagenomic_model in ner_models_metagenomics:
                    batch_annotations = batch_annotate_sentences(
                        sentences=sentences,
                        section=section_name,
                        ner_model=metagenomic_model,
                        extract_annotation_fn=extract_annotation_fn,
                    )
                    if batch_annotations:
                        all_annotations_p2.extend(batch_annotations)

            # Generate and format tags for PROVIDER_2 (if applicable)
            linked_annotations_p2 = generate_tags(
                all_annotations_p2,
                ENTITY_TYPE_MAP_2,
                linker=linker,
                use_map_terms=False,
            )
            match_json_p2, no_match_json_p2 = format_output_annotations(
                linked_annotations_p2,
                ext_id=article_data["ext_id"],
                source=article_data["source"],
                provider="metagenomics"
            )

            # Save results for PROVIDER_2
            if match_json_p2 and match_json_p2["anns"]:
                provider2_file = os.path.join(provider2_dir, os.path.basename(output_filename))
                with open(provider2_file, "a", encoding="utf-8") as file:
                    json.dump(match_json_p2, file, ensure_ascii=False)
                    file.write("\n")
            if no_match_json_p2 and no_match_json_p2["anns"]:
                no_match_file_p2 = os.path.join(provider2_dir, "no_matches", os.path.basename(output_filename))
                with open(no_match_file_p2, "a", encoding="utf-8") as file:
                    json.dump(no_match_json_p2, file, ensure_ascii=False)
                    file.write("\n")

    print(f"Processing completed. Results saved to {output_dir}")


if __name__ == '__main__':
    # Session options for ONNXRuntime
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    # Define paths
    input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch-30-10-2024-0.abstract.gz"
    output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/abstracts"
    output_fname = os.path.basename(input_path).replace(".abstract.gz", "")
    output_file = os.path.join(output_path, f"{output_fname}.json")
    ml_model_path = '/home/stirunag/work/github/CAPITAL/model/'
    article_classifier_path = os.path.join(ml_model_path, "article_classifier/")

    # Check input file and output directory
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.isdir(output_path):
        print(f"Output directory '{output_path}' does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    # Load EntityLinker with all required annotations, including primer
    print("Loading entity linker.")
    linker = EntityLinker()
    linker.load_annotations(['primer'])

    # Load PROVIDER_2 (metagenomics) NER models
    print("Loading PROVIDER_2 NER models.")
    metagenomic_paths = [os.path.join(ml_model_path, f'metagenomics/metagenomic-set-{i}_quantised') for i in range(1, 6)]
    try:
        ner_models_metagenomics = [load_ner_model(path, session_options) for path in metagenomic_paths]
    except Exception as e:
        raise RuntimeError(f"Error loading PROVIDER_2 NER models: {str(e)}")

    # Load article classifier models
    print("Loading article classifier models.")
    try:
        load_artifacts(article_classifier_path)
    except Exception as e:
        raise RuntimeError(f"Error loading article classifier models: {str(e)}")

    # Process and save articles
    print("Processing articles and saving results.")
    process_and_save_articles(
        files_list=read_file_blocks(input_path),
        extract_annotation_fn=extract_annotation,  # Assuming this is a method in EntityLinker
        classify_abstract_fn=classify_abstract,  # Replace with your classification function
        ner_models_metagenomics=ner_models_metagenomics,
        output_filename=output_file,
        output_dir=output_path,
    )

    print("Processing completed.")


    # parser = argparse.ArgumentParser(description='Process XML files and output sentences and sections.')
    # parser.add_argument('--input', help='Input XML or GZ file path', required=True)
    # parser.add_argument('--output', help='Output JSONL file path', required=True)
    # args = parser.parse_args()
    #
    # # Call process_each_article with the full output file path
    # process_each_article(args.input, args.output, args.type)






