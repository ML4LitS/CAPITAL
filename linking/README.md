
## Description

This package facilitates entity linking by using embeddings to represent terms from various knowledge bases. The core functionality includes generating embeddings, indexing these embeddings with the Facebok's FAISS library, and conducting entity searches with both semantic and fuzzy matching.

## Features

- **Embedding Generation:** Utilises floret embeddings trained on Europe PMC's open access content for contextual understanding of biomedical terms. We have also provided a notebook where we implemented the use any of huggingface models to generate embeddings.
- **FAISS Indexing:** Implements efficient vector indexing to enable rapid and precise searching.
- **Fuzzy Matching:** Enhances matching accuracy by comparing query terms against indexed terms using a fuzzy logic.

## Prerequisites

Before you begin, ensure you have the following:
- Python 3.8 or higher
- Dependencies installed from the `requirements.txt` file

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ML4LitS/CAPITAL.git
cd CAPITAL
cd linker
pip install -r requirements.txt
```
## Floret model

https://drive.google.com/drive/folders/158y1pd72iI1eXWw4gDSqHUoaJQ367vbP?usp=drive_link

## Indexed KB data

https://drive.google.com/drive/folders/1vnFBUj4dbhc6EGxObDVJpPaYo5tzqxbo?usp=drive_link

## Usage

Here's how to use the package to link entities:

1. **Load the necessary models and data:**
   ```python
   from entity_linker import EntityLinker
   linker = EntityLinker(base_path="/path/to/your/index_data")
   ```

2. **Link entities:**
   ```python
   terms = ['hypertension', 'covid-19', 'Coronavirus disease']
   annotation_type = 'DS'  # Disease
   results = linker.link_entities(terms, annotation_type)
   print(results)
   ```

## Data Structure

Ensure the following structure in your data directory for indexed embeddings and metadata:

- `chebi_terms.index`
- `chebi_terms.pkl`
- `NCBI_terms.index`
- `NCBI_terms.pkl`
- `umls_terms.index`
- `umls_terms.pkl`
- `uniprot_terms.index`
- `uniprot_terms.pkl`

## Customise for your Knowledgebase

Use the following notebook tutoriaLs to generate your custom entity linkers:

In this example we have used bio assay ontology (BAO)

Using Sentence Tranformers:
https://github.com/ML4LitS/CAPITAL/blob/main/notebooks/normalisation%20analysis/Testing%20with%20Sbert.ipynb

Using Floret Embeddings:
https://github.com/ML4LitS/CAPITAL/blob/main/notebooks/normalisation%20analysis/BAO.ipynb

## Contributing

Contributions to improve the software are welcome. Please fork the repository and submit pull requests with your enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
 
