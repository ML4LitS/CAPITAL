import faiss
import pickle
import spacy
import numpy as np
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForTokenClassification
from flask import Flask, request, jsonify, render_template
import socket
import json
import re
import string

# Initialize Flask app
app = Flask(__name__)

# Load the quantized model and tokenizer
model_path_quantised = '/home/stirunag/work/github/CAPITAL/model'
print("Loading NER model and tokenizer...")
model_quantized = ORTModelForTokenClassification.from_pretrained(
    model_path_quantised,
    file_name="model_quantized.onnx"
)
tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised, model_max_length=512, batch_size=4, truncation=True)
ner_quantized = pipeline("token-classification", model=model_quantized, tokenizer=tokenizer_quantized, aggregation_strategy="first")
print("NER model and tokenizer loaded successfully.")

# Load spaCy model
nlp = spacy.load("/home/stirunag/work/github/CAPITAL/normalisation/en_floret_model")


# Load dictionaries and initialize FAISS indexes
base_path = "/home/stirunag/work/github/CAPITAL/normalisation/dictionary/"
file_mapping = {
    'CD': ('chebi_terms.index', 'chebi_terms.pkl'),
    'OG': ('NCBI_terms.index', 'NCBI_terms.pkl'),
    'DS': ('umls_terms.index', 'umls_terms.pkl'),
    'GP': ('uniprot_terms.index', 'uniprot_terms.pkl'),
    'GO': ('go_terms.index', 'go_terms.pkl'),
    'EM': ('em_terms.index', 'em_terms.pkl')
}

# Load data for each annotation type
loaded_data = {}
print("Loading annotation data for entity linking...")
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
print("All annotation data loaded successfully.")

# Define mapping function for URLs
def mapToURL(entity_group, ent_id):
    # Return "#" if ent_id is None or empty
    if not ent_id:
        return "#"

    # Handle specific cases for entity groups
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
    else:
        # Default mapping for other entity groups
        switcher = {
            'GP': f"https://www.uniprot.org/uniprotkb/{ent_id}/entry",
            'DS': f"http://linkedlifedata.com/resource/umls-concept/{ent_id}",
            'OG': f"http://identifiers.org/taxonomy/{ent_id}",
            'CD': f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={ent_id}",
        }
        return switcher.get(entity_group, "#")


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
        match = get_exact_match(entity, term_dict)
        if not match:
            if annotation_type == 'DS':
                match = get_embedding_match(clean_term(entity.lower()), index, indexed_terms, term_dict, model)
            else:
                match = get_embedding_match(entity.lower(), index, indexed_terms, term_dict, model)
        mapped_entities[entity] = match if match else "No Match"
    return mapped_entities

# Utility functions
def clean_term(term):
    phrases_to_remove = [
        '--', 'physical finding', 'diagnosis', 'disorder', 'procedure', 'finding',
        'symptom', 'history', 'treatment', 'manifestation', 'disease', 'finding',
        'morphologic abnormality', 'etiology', 'observable entity', 'event',
        'situation', 'degrees', 'in some patients', 'cm', 'mm', '#', 'rare'
    ]
    term = term.lower()
    for phrase in phrases_to_remove:
        term = re.sub(rf'\b{re.escape(phrase)}\b', '', term, flags=re.IGNORECASE)
    term = re.sub(rf"[{string.punctuation}]", "", term)  # Remove punctuation
    return ' '.join(term.split())

def get_exact_match(term, term_dict):
    return term_dict.get(term)
#
# def get_fuzzy_match(term, term_dict, threshold=70):
#     result = process.extractOne(term, term_dict.keys(), scorer=fuzz.ratio)
#     if result and result[1] >= threshold:
#         return term_dict[result[0]]
#     return None


def is_flat_index(index):
    return isinstance(index, faiss.IndexFlat)

@app.route('/')
def index():
    return render_template('index.html', host=socket.gethostbyname(socket.gethostname()))

@app.route('/annotate_link_cli', methods=['POST'])
def annotate_link_cli():
    input_text = request.form.get('text')
    if not input_text:
        return 'No text provided', 400

    # Get NER output from the quantized model
    ner_output = ner_quantized(input_text)
    print(f"NER Output: {ner_output}")  # Debugging line

    result = [{k: round(float(v), 3) if isinstance(v, np.float32) else v for k, v in res.items()} for res in ner_output]
    x_list_ = []

    for ent in result:
        term = input_text[int(ent['start']):int(ent['end'])]
        entity_group = ent['entity_group']
        print(f"Processing entity: {term}, group: {entity_group}")  # Debugging line

        mapped_terms = map_terms([term], entity_group,nlp)
        mapped_term = mapped_terms.get(term, "No Match")
        ent_id = mapped_term if mapped_term != "No Match" else None
        url = mapToURL(entity_group, ent_id)
        x_list_.append([ent['start'], ent['end'], entity_group, mapped_term, ent['score'], ent_id, url])
        print(f"Mapped Result: {x_list_[-1]}")  # Debugging line

    # Logging
    with open('annotation_cli_log.txt', 'a') as file:
        json.dump({'Input': input_text, 'Output': x_list_}, file)
        file.write('\n')

    return jsonify(x_list_)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
