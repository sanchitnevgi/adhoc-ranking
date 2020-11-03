## Pre-requisites
This project needs Maven and Java JDK >= 11 setup (to build Anserini)

## Setup

```bash
# Create a new python environment and install dependencies
conda create --name adhoc python=3.7
conda activate adhoc

# Install requirements
pip install -r requirements.txt

# Clone and build Anserini for evaluation and baselines
# Follow the instructions from the Anserini repo here - https://github.com/castorini/anserini

# Download the data
bash ./download_data.sh

# Index TREC DLT corpus (from the Anserini root)
sh target/appassembler/bin/IndexCollection -collection CleanTrecCollection -input ../adhoc-ranking/data/ \
    -index indexes/index.msmarco-doc-generator DefaultLuceneDocumentGenerator -threads 1 \
    -storePositions -storeDocvectors -storeRaw
```
