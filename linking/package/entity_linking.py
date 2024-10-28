import faiss
import pickle
import spacy

from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity


# Load spaCy model
nlp = spacy.load("/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model")

# Define mapping of annotation type to corresponding file paths
file_mapping = {
    'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
    'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
    'DS': ('umls_terms.index', 'umls_terms.pkl'),
    'GP': ('uniprot_terms.index', 'uniprot_terms.pkl'),
    'GO': ('go_terms.index', 'go_terms.pkl'),
    'EM': ('em_terms.index', 'em_terms.pkl')
}

# Dictionary to hold the loaded data for each annotation type
loaded_data = {}

# Load all necessary files at the beginning
base_path = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/"
for annotation_type, (index_file, pkl_file) in file_mapping.items():
    with open(base_path + pkl_file, "rb") as infile:
        data = pickle.load(infile)
    index = faiss.read_index(base_path + index_file)
    loaded_data[annotation_type] = {
        "term_to_id": data["term_to_id"],
        "indexed_terms": data["indexed_terms"],
        "index": index
    }
    print(f"Loaded data for {annotation_type}")


# Functions for exact, fuzzy, and embedding-based matching
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

    # Handle search based on the type of index
    if is_flat_index(index):
        _, I = index.search(term_vector, 1)
    else:
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
        # Step 1: Initial matching
        # match = get_exact_match(entity, term_dict)
        # if not match:
        #     match = get_fuzzy_match(entity, term_dict)
        # if not match:
        match = get_embedding_match(entity, index, indexed_terms, term_dict, model)


        mapped_entities[entity] = match if match else "No Match"
    return mapped_entities


# Example terms and annotation type
terms = ['hypertension', 'covid19', 'Coronavirus', 'Diabetes Type 2']
annotation_type = 'DS'

results = map_terms(terms, annotation_type, nlp)

# Print the mapped results
print(results)


data = loaded_data[annotation_type]
term_dict = data["term_to_id"]
indexed_terms = data["indexed_terms"]
index = data["index"]


# Reverse the dictionary to map CUIs back to terms
id_to_term = {v: k for k, v in term_dict.items()}

# Check and print the term for each CUI found in the result dictionary
for term, cui in results.items():
    if cui != 'No Match':
        original_term = id_to_term.get(cui, "Unknown CUI")
        print(f"The term associated with '{cui}' is: {original_term}")