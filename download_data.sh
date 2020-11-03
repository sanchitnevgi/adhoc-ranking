# Create data folder
mkdir -p ./data

cd data
# Download Corpus
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz
# Download document lookup
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz

# Train data
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz

# Dev data
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz

# Test data
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz
wget https://trec.nist.gov/data/deep/2019qrels-docs.txt
