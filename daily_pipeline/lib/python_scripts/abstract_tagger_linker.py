import argparse
import json
import os
from bs4 import BeautifulSoup
import spacy
from entity_linker import EntityLinker
from entity_tagger import load_ner_model, batch_annotate_sentences, extract_annotation
import gzip
import re
from collections import defaultdict, OrderedDict, Counter
from tqdm import tqdm
import onnxruntime as ort
from datetime import datetime


PROVIDER = "europepmc"

# Mapping from abbreviation to full form
ENTITY_TYPE_MAP_1 = {
    "EM": "methods", #exp_methods
    "DS": "disease",
    "GP": "gene_protein",
    "GO": "go_term",
    "CD": "chemical",
    "OG": "organism"
}

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
        all_articles_list = re.findall(r"<article.*?>.*?</article>", content, re.DOTALL)
        print(len(all_articles_list))
        return all_articles_list
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return []

def extract_article_data(article_xml):
    """Parse XML of an <article> to extract relevant data."""

    try:
        soup = BeautifulSoup(article_xml, "lxml")
        ext_id = soup.find("ext_id").text if soup.find("ext_id") else None
        if not ext_id:
            print(f"Article missing ext_id, skipping. Title: {ext_id}")
            return None  # Skip this article

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
        title = soup.find("title").text if soup.find("title") else "No Title"
        return None

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
    ner_model,
    extract_annotation_fn,
    output_filename,
    output_dir
    ):
    """
    Process articles, saving results immediately.

    Args:
        files_list (list): List of raw article XML strings.
        ner_model: Named entity recognition model for PROVIDER_1.
        extract_annotation_fn (callable): Function to extract annotations.
        output_dir (str): Directory to save results.

    Returns:
        None
    """
    provider1_dir = os.path.join(output_dir, PROVIDER)
    os.makedirs(provider1_dir, exist_ok=True)
    os.makedirs(os.path.join(provider1_dir, "no_matches"), exist_ok=True)


    for article_xml in tqdm(files_list, desc="Processing Articles"):
        article_data = extract_article_data(article_xml)
        if not article_data or "sections" not in article_data:
            print(f"Skipping malformed article: {article_xml[:100]}...")
            continue

        ext_id = article_data["ext_id"]

        ### Process PROVIDER_1 ###
        all_annotations_p1 = []
        for section_name, sentences in article_data["sections"].items():
            batch_annotations = batch_annotate_sentences(
                sentences=sentences,
                section=section_name,
                ner_model=ner_model,
                extract_annotation_fn=extract_annotation_fn,
            )
            if batch_annotations:
                all_annotations_p1.extend(batch_annotations)

        # Generate and format tags for PROVIDER_1
        linked_annotations_p1 = generate_tags(
            all_annotations_p1,
            ENTITY_TYPE_MAP_1,
            linker=linker
        )
        match_json_p1, no_match_json_p1 = format_output_annotations(
            linked_annotations_p1,
            ext_id=article_data["ext_id"],
            source=article_data["source"],
            provider="europepmc"
        )

        # Save results for PROVIDER_1
        if match_json_p1 and match_json_p1["anns"]:
            provider1_file = os.path.join(provider1_dir, os.path.basename(output_filename))
            with open(provider1_file, "a", encoding="utf-8") as file:
                json.dump(match_json_p1, file, ensure_ascii=False)
                file.write("\n")
        if no_match_json_p1 and no_match_json_p1["anns"]:
            no_match_file_p1 = os.path.join(provider1_dir, "no_matches", os.path.basename(output_filename))
            with open(no_match_file_p1, "a", encoding="utf-8") as file:
                json.dump(no_match_json_p1, file, ensure_ascii=False)
                file.write("\n")


    print(f"Processing completed. Results saved to {output_dir}")


def generate_output_basename(input_basename):
    """
    Generate an output basename by reformatting the date from the input basename.

    Args:
        input_basename (str): The input filename (e.g., 'patch-28-10-2024-0.abstract.gz').

    Returns:
        str: The output filename (e.g., 'patch-2024_10_28-0.api.json').

    Raises:
        ValueError: If the date in the input basename does not match the expected format.
    """
    try:
        # Regular expression to match the expected input structure
        pattern = r'^patch-(\d{1,2}-\d{1,2}-\d{4})-(\d+)\.abstract\.gz$'
        match = re.match(pattern, input_basename)

        if not match:
            raise ValueError(f"Input basename does not match the expected format: {input_basename}")

        # Extract the date and index parts
        date_str, index_str = match.groups()

        # Convert the date to YYYY_MM_DD format
        formatted_date = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y_%m_%d")

        # Construct the output basename
        output_basename = f"patch-{formatted_date}-{index_str}"

        return output_basename

    except Exception as e:
        raise ValueError(f"Error processing input basename '{input_basename}': {e}")


if __name__ == '__main__':
    # Session options for ONNXRuntime
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    # Define paths
    input_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/notebooks/data/patch-28-10-2024-0.abstract.gz"
    output_path = "/home/stirunag/work/github/CAPITAL/daily_pipeline/results/abstracts"

    input_basename = os.path.basename(input_path)
    output_fname = generate_output_basename(input_basename)
    output_file = os.path.join(output_path, f"{output_fname}.api.json")



    # output_fname = os.path.basename(input_path).replace(".abstract.gz", "")
    # output_file = os.path.join(output_path, f"{output_fname}.json")
    model_path_quantised = "/home/stirunag/work/github/CAPITAL/model/europepmc"
    ml_model_path = '/home/stirunag/work/github/CAPITAL/model/'

    # Check input file and output directory
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.isdir(output_path):
        print(f"Output directory '{output_path}' does not exist. Creating it.")
        os.makedirs(output_path, exist_ok=True)

    # Load PROVIDER_1 (europepmc) NER model
    print("Loading PROVIDER_1 NER model.")
    ner_quantized = load_ner_model(model_path_quantised, session_options)

    # Load EntityLinker with all required annotations, including primer
    print("Loading entity linker.")
    linker = EntityLinker()
    linker.load_annotations(['EM', 'DS', 'GP', 'GO', 'CD', 'OG'])

    # Process and save articles
    print("Processing articles and saving results.")
    process_and_save_articles(
        files_list=read_file_blocks(input_path),
        ner_model=ner_quantized,
        extract_annotation_fn=extract_annotation,  # Assuming this is a method in EntityLinker
        output_filename=output_file,
        output_dir=output_path
    )

    print("Processing completed.")

#################################################################################################################
    # parser = argparse.ArgumentParser(description='Process abstract patch XML files and output jsonl api.')
    # parser.add_argument('--input', help='Input XML or GZ file path', required=True)
    # parser.add_argument('--output', help='Output JSONL file path', required=True)
    # parser.add_argument('--model_path', help='Path to the quantized model directory', required=True)
    # args = parser.parse_args()
    # #
    #
    # input_path = args.input
    # output_path = args.output
    # model_path_quantised = args.model_path
    #
    # input_basename = os.path.basename(input_path)
    # # Replace ".abstract.gz" and convert date to YYYY_MM_DD format
    # patch_part, date_str, index_part = input_basename.rsplit("-", 2)
    # formatted_date = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y_%m_%d")
    # index_str = index_part.replace(".abstract.gz", "")
    # output_fname = f"{patch_part}-{formatted_date}-{index_str}"
    # # output_file = os.path.join(output_path, f"{output_fname}.api.json")
    # output_file = os.path.join(output_path, f"{output_fname}.api.json")
    #
    # # Check input file and output directory
    # if not os.path.isfile(input_path):
    #     raise FileNotFoundError(f"Input file not found: {input_path}")
    # if not os.path.isdir(output_path):
    #     print(f"Output directory '{output_path}' does not exist. Creating it.")
    #     os.makedirs(output_path, exist_ok=True)
    #
    # # Load PROVIDER_1 (europepmc) NER model
    # print("Loading PROVIDER_1 NER model.")
    # # Define or import session_options before using
    # ner_quantized = load_ner_model(model_path_quantised, session_options)
    #
    # # Load EntityLinker with all required annotations, including primer
    # print("Loading entity linker.")
    # linker = EntityLinker()
    # linker.load_annotations(['EM', 'DS', 'GP', 'GO', 'CD', 'OG'])
    #
    # # Process and save articles
    # print("Processing articles and saving results.")
    # process_and_save_articles(
    #     files_list=read_file_blocks(input_path),
    #     ner_model=ner_quantized,
    #     extract_annotation_fn=extract_annotation,  # Correctly reference the method
    #     output_filename=output_file,
    #     output_dir=output_path
    # )
    #
    # print("Processing completed.")
    #






