import faiss
import pickle
import spacy
# import numpy as np
import re
import sys
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
import string

# from flask_GUI.flask_app import loaded_data

# import pandas as pd

# Load spaCy model
MODEL_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model"
try:
    nlp = spacy.load(MODEL_PATH)
except OSError:
    print(f"Error: The spaCy model at '{MODEL_PATH}' was not found.", file=sys.stderr)
    sys.exit(1)

# Mapping of file names to each annotation type
FILE_MAPPING = {
    'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
    'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
    'DS': ('umls_terms.index', 'umls_terms.pkl'),
    'GP': ('uniprot_terms.index', 'uniprot_terms.pkl'),
    'GO': ('go_terms.index', 'go_terms.pkl'),
    'EM': ('em_terms.index', 'em_terms.pkl'),
    'primer': ('primer_terms.index', 'primer_terms.pkl')  # Added comma
}

# Load data and indices for each annotation type
BASE_PATH = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/"

def load_annotations(annotation_types=None):
    """
    Load specified annotation types or all if none are specified.

    Args:
        annotation_types (list or None): List of annotation type keys to load.
                                         If None, all annotation types are loaded.

    Returns:
        dict: A dictionary containing loaded data for each annotation type.
              The structure is:
              {
                  'AnnotationType': {
                      'term_to_id': {...},
                      'indexed_terms': ...,
                      'index': faiss.Index,
                      'id_to_term': {...}
                  },
                  ...
              }
    """
    loaded_data = {}

    # If no specific annotation types are provided, load all
    if annotation_types is None:
        annotation_types = list(FILE_MAPPING.keys())
        print("No specific annotation types provided. Loading all annotation types.")
    else:
        # Validate provided annotation types
        invalid_types = [atype for atype in annotation_types if atype not in FILE_MAPPING]
        if invalid_types:
            print(f"Error: The following annotation types are invalid: {', '.join(invalid_types)}", file=sys.stderr)
            print(f"Valid annotation types are: {', '.join(FILE_MAPPING.keys())}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading specified annotation types: {', '.join(annotation_types)}")

    for annotation_type in annotation_types:
        index_file, pkl_file = FILE_MAPPING[annotation_type]

        # Load pickle file
        try:
            with open(BASE_PATH + pkl_file, "rb") as infile:
                data = pickle.load(infile)
        except FileNotFoundError:
            print(f"Error: Pickle file '{pkl_file}' for annotation type '{annotation_type}' not found.", file=sys.stderr)
            continue
        except pickle.UnpicklingError:
            print(f"Error: Failed to unpickle file '{pkl_file}' for annotation type '{annotation_type}'.", file=sys.stderr)
            continue

        # Load FAISS index
        try:
            index = faiss.read_index(BASE_PATH + index_file)
        except Exception as e:
            print(f"Error: Failed to load FAISS index '{index_file}' for annotation type '{annotation_type}'.\n{e}", file=sys.stderr)
            continue

        # Create a reverse mapping from term_id to term
        id_to_term = {v: k for k, v in data["term_to_id"].items()}

        # Store loaded data
        loaded_data[annotation_type] = {
            "term_to_id": data["term_to_id"],
            "indexed_terms": data.get("indexed_terms", None),  # Handle if 'indexed_terms' is not present
            "index": index,
            "id_to_term": id_to_term
        }

        print(f"Loaded data for '{annotation_type}'.")

    return loaded_data

# List of phrases to remove (converted to lowercase)
phrases_to_remove = [
    '--', 'physical finding', 'diagnosis', 'disorder', 'procedure', 'finding',
    'symptom', 'history', 'treatment', 'manifestation', 'disease', 'finding',
    'morphologic abnormality', 'etiology', 'observable entity', 'event',
    'situation', 'degrees', 'in some patients', 'cm', 'mm',
    '#', 'rare', 'degree', 'including anastomotic', 'navigational concept',
    '1 patient', 'qualifier value', 'lab test', 'unintentional',
    'tophi', 'nos', 'msec', 'reni', 'less common', 'as symptom'
]


# Function to clean term
def clean_term(term):
    # Convert term to lowercase for consistent comparison
    term_lower = term.lower()

    # Remove specified phrases
    for phrase in phrases_to_remove:
        term_lower = re.sub(rf'\b{re.escape(phrase)}\b', '', term_lower, flags=re.IGNORECASE)

    # Remove punctuation
    term_cleaned = term_lower.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    term_cleaned = ' '.join(term_cleaned.split())

    return term_cleaned


# Function to clean term
def clean_term_EM(term):
    # Convert term to lowercase for consistent comparison
    if term.endswith("es"):
        return term[:-2]
    elif term.endswith("s"):
        return term[:-1]
    return term

    return term_cleaned


# Matching functions
def get_exact_match(term, term_dict):
    return term_dict.get(term)


def get_fuzzy_match(term, term_dict, threshold=70):
    result = process.extractOne(term, term_dict.keys(), scorer=fuzz.ratio)
    if result:
        match, score = result[0], result[1]
        if score >= threshold:
            return term_dict[match]
    return None


def is_flat_index(index):
    return isinstance(index, faiss.IndexFlat)


def get_embedding_match(term, index, indexed_terms, term_dict, model, threshold=0.7):
    term_vector = model(term).vector.reshape(1, -1).astype('float32')
    faiss.normalize_L2(term_vector)

    # Perform search on the FAISS index
    _, I = index.search(term_vector, 1)

    if I[0][0] != -1:
        matched_term = indexed_terms[I[0][0]]
        similarity = cosine_similarity(term_vector, model(matched_term).vector.reshape(1, -1))[0][0]
        if similarity >= threshold:
            return term_dict.get(matched_term, "No Match")
    return None



def map_terms(entities, annotation_type, model):
    """Map new entities using exact, fuzzy, and embedding matches, with abbreviation fallback."""
    data = loaded_data[annotation_type]
    term_dict = data["term_to_id"]
    indexed_terms = data["indexed_terms"]
    index = data["index"]

    mapped_entities = {}
    for entity in entities:
        match = get_exact_match(entity, term_dict)
        if not match:
            if annotation_type == 'DS':
                match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
            elif annotation_type == 'EM':
                match = get_embedding_match(clean_term_EM(entity.lower()), index, indexed_terms, term_dict, model)
            else:
                match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)
        mapped_entities[entity] = match if match else "No Match"
    return mapped_entities

def map_terms_reverse(entities, annotation_type, model):
    """Map entities using exact, similarity, and embedding matches, returning both code and term."""
    data = loaded_data[annotation_type]
    term_dict = data["term_to_id"]
    id_to_term = data["id_to_term"]
    indexed_terms = data["indexed_terms"]
    index = data["index"]

    mapped_entities = {}
    for entity in entities:
        # Normalize entity based on annotation type requirements
        normalized_entity = entity.lower() if annotation_type != 'GP' else entity

        # Try exact match
        match = get_exact_match(normalized_entity, term_dict)

        # If exact match fails, try similarity or embedding matching
        if not match:
            if annotation_type == 'DS':
                match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
            elif annotation_type == 'EM':
                match = get_embedding_match(clean_term_EM(entity.lower()), index, indexed_terms, term_dict, model)
            else:
                match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)

        # Set grounded_code and grounded_term based on match
        if match:
            grounded_code = match
            grounded_term = id_to_term.get(match, "Unknown Term")  # Retrieve term from id_to_term or use default
        else:
            grounded_code, grounded_term = "No Match", "No Match"

        # Ensure both code and term are consistently stored in the result
        mapped_entities[entity] = (grounded_code, grounded_term)

    return mapped_entities


# Example terms and annotation type
loaded_data = load_annotations(annotation_types=['primer'])
# terms = ['hypertension', 'covid19', 'Coronavirus', 'Diabetes Type 2', 'abdomenal HeRNIA!!']
annotation_type = 'primer'
terms = ['E166', 'Fmoc-Leu-Val-D-Leu-O-Resin', 'U1510R', 'Fmoc-Leu-', 'OTU25to31-1406']

results = map_terms(terms, annotation_type, nlp)
# Print the mapped results
print(results)

reverse_result = map_terms_reverse(terms, annotation_type, nlp)
print(reverse_result)
