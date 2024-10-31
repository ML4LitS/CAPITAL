import faiss
import pickle
import spacy
import re
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
import string
import os
from dotenv import load_dotenv

# Load .env_paths using the relative path from python_scripts
# load_dotenv('../../.env_paths')
load_dotenv('/home/stirunag/work/github/CAPITAL/daily_pipeline/.env_paths')

# Load paths from .env_paths
spacy_model_path = os.getenv("SPACY_MODEL_PATH")
base_dictionary_path = os.getenv("BASE_DICTIONARY_PATH")

print(spacy_model_path)
print(base_dictionary_path)

# Verify paths are loaded correctly
if not spacy_model_path or not base_dictionary_path:
    raise ValueError("SPACY_MODEL_PATH and BASE_DICTIONARY_PATH must be set in .env_paths")

# Load spaCy model
print("Loading spaCy model for entity linking...")
model = spacy.load(spacy_model_path)
print("SpaCy model loaded successfully.")

# Mapping of file names to each annotation type
file_mapping = {
    'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
    'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
    'DS': ('umls_terms.index', 'umls_terms.pkl'),
    'GP': ('uniprot_terms.index', 'uniprot_terms.pkl'),
    'GO': ('go_terms.index', 'go_terms.pkl'),
    'EM': ('em_terms.index', 'em_terms.pkl')
}

# Load data and indices for each annotation type

loaded_data = {}
for annotation_type, (index_file, pkl_file) in file_mapping.items():
    with open(base_dictionary_path + pkl_file, "rb") as infile:
        data = pickle.load(infile)
    index = faiss.read_index(base_dictionary_path + index_file)

    # Create a reverse mapping from CUI to term
    id_to_term = {v: k for k, v in data["term_to_id"].items()}

    # Store loaded data
    loaded_data[annotation_type] = {
        "term_to_id": data["term_to_id"],
        "indexed_terms": data["indexed_terms"],
        "index": index,
        "id_to_term": id_to_term
    }
    print(f"Loaded data for {annotation_type}")

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


def map_terms(entities, entity_type):
   """Map new entities using exact, fuzzy, and embedding matches, with abbreviation fallback."""
   data = loaded_data[entity_type]
   term_dict = data["term_to_id"]
   indexed_terms = data["indexed_terms"]
   index = data["index"]

   mapped_entities = {}
   for entity in entities:
       match = get_exact_match(entity, term_dict)
       if not match:
           if entity_type == 'DS':
               match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
           elif entity_type == 'EM':
               match = get_embedding_match(clean_term_EM(entity.lower()), index, indexed_terms, term_dict, model)
           else:
               match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)
       mapped_entities[entity] = match if match else "No Match"
   return mapped_entities

def map_terms_batch(all_entities):
    """
    Map entities for each entity type using exact, fuzzy, and embedding matches,
    with an abbreviation fallback if no matches are found.
    """
    mapped_entities = {}
    unmatched_terms = []

    for entity_type, entities in all_entities.items():
        data = loaded_data[entity_type]
        term_dict = data["term_to_id"]
        indexed_terms = data["indexed_terms"]
        index = data["index"]

        for entity in entities:
            match = get_exact_match(entity, term_dict)
            if not match:
                if entity_type == 'DS':
                    match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
                elif entity_type == 'EM':
                    match = get_embedding_match(clean_term_EM(entity.lower()), index, indexed_terms, term_dict, model)
                else:
                    match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)

            # Only add to mapped_entities if there's a match
            if match:
                mapped_entities[entity] = match
            else:
                unmatched_terms.append({"term": entity})

    return mapped_entities, unmatched_terms


def map_terms_reverse(entities, entity_type):
    """Map entities using exact, similarity, and embedding matches, returning both code and term."""
    data = loaded_data[entity_type]
    term_dict = data["term_to_id"]
    id_to_term = data["id_to_term"]
    indexed_terms = data["indexed_terms"]
    index = data["index"]

    mapped_entities = {}
    for entity in entities:
        # Normalize entity based on annotation type requirements
        normalized_entity = entity.lower() if entity_type != 'GP' else entity

        # Try exact match
        match = get_exact_match(normalized_entity, term_dict)

        # If exact match fails, try similarity or embedding matching
        if not match:
            if entity_type == 'DS':
                match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
            elif entity_type == 'EM':
                match = get_embedding_match(clean_term_EM(entity.lower()), index, indexed_terms, term_dict, model)
            else:
                match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)

        # Set grounded_code and grounded_term based on match
        if match:
            grounded_code = match
            grounded_term = id_to_term.get(match, "#")  # Retrieve term from id_to_term or use default
        else:
            grounded_code, grounded_term = "#", "#"

        # Ensure both code and term are consistently stored in the result
        mapped_entities[entity] = (grounded_code, grounded_term)

    return mapped_entities

def map_terms_reverse_batch(all_entities):
    """
    Map entities using exact, similarity, and embedding matches, returning both code and term.
    Returns mapped entities and a list of unmatched terms.
    """
    mapped_entities = {}
    unmatched_terms = []

    for entity_type, entities in all_entities.items():
        data = loaded_data[entity_type]
        term_dict = data["term_to_id"]
        id_to_term = data["id_to_term"]
        indexed_terms = data["indexed_terms"]
        index = data["index"]

        for entity in entities:
            # Normalize entity based on annotation type requirements
            normalized_entity = entity.lower() if entity_type != 'GP' else entity

            # Try exact match
            match = get_exact_match(normalized_entity, term_dict)

            # If exact match fails, try similarity or embedding matching
            if not match:
                if entity_type == 'DS':
                    match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
                elif entity_type == 'EM':
                    match = get_embedding_match(clean_term_EM(entity.lower()), index, indexed_terms, term_dict, model)
                else:
                    match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)

            # Set grounded_code and grounded_term based on match
            if match:
                grounded_code = match
                grounded_term = id_to_term.get(match, "Unknown Term")
                mapped_entities[entity] = (grounded_code, grounded_term)
            else:
                # If no match, add entity to unmatched_terms
                unmatched_terms.append({"term": entity})

    return mapped_entities, unmatched_terms


def map_to_url(entity_group, ent_id):
    # Add the specialized mapping logic
    if not ent_id:
        return "#"
    if entity_group == 'EM':
        if ent_id.startswith("EFO"):
            return f"http://www.ebi.ac.uk/efo/{ent_id}"
        elif ent_id.startswith("MI") or ent_id.startswith("OBI"):
            return f"http://purl.obolibrary.org/obo/{ent_id}"
        else:
            return "#"
    elif entity_group == 'GO':
        ent_id = ent_id.replace('_', ':')
        return f"http://identifiers.org/go/{ent_id}"
    # Default mappings
    url_map = {
        'GP': f"https://www.uniprot.org/uniprotkb/{ent_id}/entry",
        'DS': f"http://linkedlifedata.com/resource/umls-concept/{ent_id}",
        'OG': f"http://identifiers.org/taxonomy/{ent_id}",
        'CD': f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={ent_id}"
    }
    return url_map.get(entity_group, "#")



