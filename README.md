# CAPITAL: Contextual Annotations PIpeline for TAgging and Linking

## Replace dictionary text mining in Europe PMC with Machine Learning based annotations

The project aims to replace the existing dictionary text mining approach with a machine learning-based solution. 
This shift is expected to improve the efficiency and accuracy of information extraction from resources. 
The project is divided into several tasks to be completed over multiple phases.

By replacing the existing dictionary text mining with machine learning, the project aims to enhance the overall efficiency, accuracy, and effectiveness of information extraction from resources. 
The successful implementation of this project is expected to have a positive impact on the Europe PMCs data processing capabilities and ultimately improve the quality of the Annotations.


### Objectives:
1. Implemement deep learning models for entity recognition
	The entity tagging project is available at https://github.com/ML4LitS/annotation_models
2. Implement entity linking tool which links the entities recognised to a knowledgebase.
	This repository aims to achieve this objective
	

This repository presents a linking strategy for various biological and medical knowledge bases, including UMLS (Diseases), UniProt (Genes/Proteins), NCBI Taxonomy (Organisms), and ChEBI (Chemicals).

## Description
Our strategy extracts terms and IDs from the provided knowledge bases and integrates them into a unified dictionary. This allows for easier cross-referencing and searching across various datasets.

## Features
UMLS Linking: Extract terms and IDs related to diseases.
UniProt Linking: Extract terms and IDs related to genes and proteins.
NCBI Taxonomy Linking: Extract terms and IDs for organisms.
ChEBI Linking: Extract terms and IDs for chemicals.

## Installation

## Requirements
Python 3.7+
Required libraries (these can be installed using pip):
tqdm
[Other required libraries...]

## Steps
Clone the repository:

git clone [Repository URL]
cd [Repository Name]

### Install the required libraries:

pip install -r requirements.txt




## Documentation
Detailed documentation for each module can be found here.

## Contributing
We welcome contributions! Please read our CONTRIBUTING guide to get started.

## License
[CC-by 4]


 
