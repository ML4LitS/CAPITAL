import faiss
import pickle
import spacy
import numpy as np
from fuzzywuzzy import fuzz

# Load spaCy model
nlp = spacy.load("/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model")

# Define mapping of annotation type to corresponding file paths
file_mapping = {
    'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
    'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
    'DS': ('umls_terms.index', 'umls_terms.pkl'),
    'GP': ('uniprot_terms.index', 'uniprot_terms.pkl')
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
        "index": index,
        "indexed_terms_ids": [(term, data["term_to_id"][term]) for term in data["indexed_terms"]]
    }
    print(f"Loaded data for {annotation_type}")

def get_average_embeddings_batched(terms):
    docs = list(nlp.pipe(terms))
    embeddings = []
    for doc in docs:
        valid_vectors = [token.vector for token in doc if token.has_vector and token.vector_norm != 0 and token.vector.shape[0] == 300]
        embeddings.append(np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros((300,)))
    return embeddings

# def retrieve_similar_terms_with_fuzzy_batched(terms, annotation_type, k=3):
#     data = loaded_data[annotation_type]
#     term_to_id, indexed_terms, index, indexed_terms_ids = data["term_to_id"], data["indexed_terms"], data["index"], data["indexed_terms_ids"]
#
#     term_embeddings = get_average_embeddings_batched(terms)
#     normalized_embeddings = [emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb for emb in term_embeddings]
#     D, I = index.search(np.array(normalized_embeddings).astype('float32'), k)
#
#     results = {}
#     for idx, term in enumerate(terms):
#         candidate_terms_and_ids = [indexed_terms_ids[i] for i in I[idx]]
#         candidate_terms, candidate_ids = zip(*candidate_terms_and_ids)
#         scores = [fuzz.ratio(term, c_term) for c_term in candidate_terms]
#         results[term] = sorted(list(zip(candidate_terms, scores, candidate_ids)), key=lambda x: x[1], reverse=True)[:k]
#
#     return results

def retrieve_similar_terms_with_fuzzy_batched(terms, annotation_type, k=3):
    data = loaded_data[annotation_type]
    term_to_id, indexed_terms, index, indexed_terms_ids = data["term_to_id"], data["indexed_terms"], data["index"], data["indexed_terms_ids"]

    # Map transformed terms to original terms
    original_to_transformed = {}
    transformed_terms = []

    # Check for entity groups that need transformation
    for term in terms:
        if annotation_type in ['CD', 'OG', 'DS']:
            transformed_term = term.lower()
            original_to_transformed[transformed_term] = term
            transformed_terms.append(transformed_term)
        else:
            original_to_transformed[term] = term
            transformed_terms.append(term)

    term_embeddings = get_average_embeddings_batched(transformed_terms)
    normalized_embeddings = [emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb for emb in term_embeddings]
    D, I = index.search(np.array(normalized_embeddings).astype('float32'), k)

    results = {}
    for idx, transformed_term in enumerate(transformed_terms):
        original_term = original_to_transformed[transformed_term]
        candidate_terms_and_ids = [indexed_terms_ids[i] for i in I[idx]]
        candidate_terms, candidate_ids = zip(*candidate_terms_and_ids)
        scores = [fuzz.ratio(transformed_term, c_term.lower()) for c_term in candidate_terms]
        results[original_term] = sorted(list(zip(candidate_terms, scores, candidate_ids)), key=lambda x: x[1], reverse=True)[:k]

    return results




terms = ['hypertension', 'covid-19', 'Coronavirus disease ']
annotation_type = 'DS'
results = retrieve_similar_terms_with_fuzzy_batched(terms, annotation_type, k=2)
print(results)




# import faiss
# import pickle
# import spacy
# import numpy as np
# from fuzzywuzzy import fuzz
#
# # Load spaCy model
# nlp = spacy.load("/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model")
#
# # Define mapping of annotation type to corresponding file paths
# file_mapping = {
#     'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
#     'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
#     'DS': ('umls_terms.index', 'umls_terms.pkl'),
#     'GP': ('uniprot_terms.index', 'uniprot_terms.pkl')
# }
#
# # Dictionary to hold the loaded data for each annotation type
# loaded_data = {}
#
# # Load all necessary files at the beginning
# base_path = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/"
# for annotation_type, (index_file, pkl_file) in file_mapping.items():
#     with open(base_path + pkl_file, "rb") as infile:
#         data = pickle.load(infile)
#     index = faiss.read_index(base_path + index_file)
#     loaded_data[annotation_type] = {
#         "term_to_id": data["term_to_id"],
#         "indexed_terms": data["indexed_terms"],
#         "index": index,
#         "indexed_terms_ids": [(term, data["term_to_id"][term]) for term in data["indexed_terms"]]
#     }
#     print(f"Loaded data for {annotation_type}")
#
# def get_average_embeddings_batched(terms):
#     docs = list(nlp.pipe(terms))
#     embeddings = []
#     for doc in docs:
#         valid_vectors = [token.vector for token in doc if token.has_vector and token.vector_norm != 0 and token.vector.shape[0] == 300]
#         embeddings.append(np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros((300,)))
#     return embeddings
#
# def retrieve_similar_terms_with_fuzzy_batched(terms, annotation_type, k=10):
#     data = loaded_data[annotation_type]
#     term_to_id, indexed_terms, index, indexed_terms_ids = data["term_to_id"], data["indexed_terms"], data["index"], data["indexed_terms_ids"]
#
#     term_embeddings = get_average_embeddings_batched(terms)
#     normalized_embeddings = [emb / np.linalg.norm(emb) if np.linalg.norm(emb) != 0 else emb for emb in term_embeddings]
#     D, I = index.search(np.array(normalized_embeddings).astype('float32'), k)
#
#     results = {}
#     for idx, term in enumerate(terms):
#         candidate_terms_and_ids = [indexed_terms_ids[i] for i in I[idx]]
#         candidate_terms, candidate_ids = zip(*candidate_terms_and_ids)
#         scores = [fuzz.ratio(term, c_term) for c_term in candidate_terms]
#         results[term] = sorted(list(zip(candidate_terms, scores, candidate_ids)), key=lambda x: x[1], reverse=True)[:k]
#
#     return results
#
#
#
# terms = ['bacteria', 'ncov-19']
# annotation_type = 'OG'
# retrieve_similar_terms_with_fuzzy_batched(terms, annotation_type, k=10)