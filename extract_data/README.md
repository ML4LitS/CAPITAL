
# Repository Overview

This repository contains two Python scripts designed to interface with the Europe PMC Annotations API and the Europe PMC RESTful API. These scripts are used for the following purposes:

1. **Annotations Extraction**: This script fetches accession numbers by type from the Europe PMC Annotations API and writes them to a CSV file.
2. **Abstracts Retrieval**: This script retrieves abstracts for specified PMC and MED IDs from the Europe PMC RESTful API and writes them to another CSV file.

## Getting Started

### Prerequisites

Before running these scripts, ensure you have the following installed:
- Python 3.6 or higher
- `requests` library
- `tqdm` library
- `csv` module (standard library)

### Installation

To get started with these scripts, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ML4LitS/CAPITAL.git
cd CAPITAL/extract_data
pip install requests tqdm
```

## File Structure

1. **types.txt**: This file should contain the types of accession numbers you want to process, one per line.
2. **final_IDs_100000.csv**: This file should list the PMC and MED IDs for which you want to fetch abstracts.

## Usage

### Annotations Extraction Script

This script reads the types of accession numbers from `types.txt`, queries the API, and appends the results to `output.csv`. It skips types already processed.

1. Ensure `types.txt` is in the same directory as the script.
2. Run the script:

```bash
python annotations_extraction.py
```

### Abstracts Retrieval Script

This script reads PMC and MED IDs from `final_IDs_100000.csv`, fetches their abstracts from the Europe PMC RESTful API, and writes them to `final_abstracts_100000.csv`.

1. Ensure `final_IDs_100000.csv` is in the same directory as the script.
2. Run the script:

```bash
python abstracts_retrieval.py
```

## API Information

### API Limits

- **Articles API**:
  - 10 requests per second
  - 500 requests per minute

- **Annotations API**:
  - Limited to 8 articles per page of results

### Types of Accession Numbers

The script can extract various types of accession numbers including but not limited to: alphafold, arrayexpress, ensembl, uniprot, wormbase, and more.

### ID Formats

Examples of ID formats in the input file:
- alphafold,MED,37764423
- alphafold,PMC,PMC10547965
- arrayexpress,MED,16672618
- arrayexpress,MED,26279681

### Sources

IDs can come from different sources such as:
- MED - PubMed / MEDLINE
- PMC - PubMed Central
- PPR - Preprints
- AGR - Agricola
- NBK - NCBI bookshelf

### API Response Formats

- **Articles API**: XML (default), JSON, Dublin Core (search method only)
- **Annotations API**: JSON (default), XML, JSON-LD (JSON Linked Data), ID_LIST (list of article identifiers)

### Common Issues

- **API Limits**: The Europe PMC APIs may have request limits. If you encounter any rate limiting issues, consider adding delays between requests or contacting Europe PMC for an increased quota.
- **File Not Found**: Make sure all required files are in the correct directory before running the scripts.
- **Network Issues**: Ensure your machine has a stable internet connection to avoid disruptions during API interactions.

## Support

For any issues or questions, please open an issue in this GitHub repository.
