import argparse
import os
import gzip
import csv
import logging
import random
from collections import defaultdict

import torch

from finetune_lf import RankingModel

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def get_doc_offset(data_dir):
    docoffset = {}

    doc_lookup_path = os.path.join(data_dir, "msmarco-docs-lookup.tsv.gz")
    with gzip.open(doc_lookup_path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)
    
    return docoffset

def get_content(docid, f):
    """get_content(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])

    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    
    return line.rstrip()

# For test set
# Create output -> qid, "Q0", docid, rank, score, run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True, type=str, help="Path to the data")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the model checkpoint")

    args = parser.parse_args()

    # Load model from checkpoint
    model = RankingModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()
    logging.info(f"Loaded model from checkpoint '{args.checkpoint}'")

    # Read the test queries (qid, query)
    test_query_path = os.path.join(args.data_dir, "msmarco-test2019-queries.tsv.gz")
    assert os.path.exists(test_query_path), f"'{test_query_path}' not found"
    
    with gzip.open(test_query_path, 'rt', encoding='utf8') as q:
        queries = list(csv.reader(q, delimiter="\t"))

    # Log query stats and sample
    logging.info(f"Total Test Queries: {len(queries)}")
    logging.info(f"Sample query: {random.choice(queries)}")

    # Map q_id to query
    query_map = {q_id: query for q_id, query in queries }

    # In the corpus tsv, each docid occurs at offset docoffset[docid]
    docoffset = get_doc_offset(args.data_dir)
    logging.info(f"Built document offset table")

    top_100_doc_path = os.path.join(args.data_dir, "msmarco-doctest2019-top100.gz")
    assert os.path.exists(top_100_doc_path), f"'{test_query_path}' not found"
    
    # Stores q_id -> [(doc_id, score)]
    query_doc_scores = defaultdict(list)

    # TODO: Read from model params
    max_seq_length = 2048
    
    # For each qid get the top100 documents
    with gzip.open(top_100_doc_path, 'rt', encoding='utf8') as t:
        for line in t:
            q_id, _, doc_id, rank, base_score, _ = line.split()
            # Get the document content
            body = get_content(doc_id, t)

            # Tokenize the query doc pair, build global attention mask
            inputs = model.tokenizer(
                query_map[q_id], body, 
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True
            )

            EOS_TOKEN = 2
            global_attention_mask = torch.zeros(max_seq_length)
            global_attention_mask[:inputs["input_ids"].index(EOS_TOKEN)] = 1

            loss, logits = model.forward(
                inputs["input_ids"].unsqueeze(0), 
                inputs["attention_mask"].unsqueeze(0), 
                global_attention_mask.unsqueeze(0),
                torch.ones(1)
            )
    
    # Sort by the score
    # Write to file in TREC format