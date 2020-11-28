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

# Run a fine-tuned BM25 model
target/appassembler/bin/SearchCollection -index indexes/lucene-index.msmarco-doc.pos+docvectors+raw \
 -topicreader TsvInt -topics src/main/resources/topics-and-qrels/topics.dl19-doc.txt \
 -bm25 -bm25.k1 3.44 -bm25.b 0.87 -output run.dl19-doc.bm25-tuned.topics.dl19-doc.txt

# Evaluate the result
eval/trec_eval.9.0.4/trec_eval -m map -c -m ndcg_cut.10 -c -m recip_rank -c -m recall.100 -c -m recall.1000 -c src/main/resources/topics-and-qrels/qrels.dl19-doc.txt run.dl19-doc.bm25-tuned.topics.dl19-doc.txt

# Run fine-tuned BM25 + RM3 model
target/appassembler/bin/SearchCollection -index indexes/lucene-index.msmarco-doc.pos+docvectors+raw \
 -topicreader TsvInt -topics src/main/resources/topics-and-qrels/topics.dl19-doc.txt \
 -bm25 -bm25.k1 3.44 -bm25.b 0.87 -rm3 -output run.dl19-doc.bm25-tuned+rm3.topics.dl19-doc.txt
```
## Commands

```python
df = pd.read_csv("./triples.tsv", sep="\t", names=["topicid", "qy", "posdocid", "posurl", "postitle", "posbody", "rnddocid", "rndurl", "rndtitle", "rndbody"])
```