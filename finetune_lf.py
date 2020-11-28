import argparse
import logging
import csv
import os
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AdamW,
    LongformerModel,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    get_linear_schedule_with_warmup
)

from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

QueryTriple = namedtuple("QueryTriple", ["topic_id", "query", "rel_doc_id", "rel_doc_url", "rel_doc_title", "rel_doc_body", 
"rnd_doc_id", "rnd_doc_url", "rnd_doc_title", "rnd_doc_body"])

class RankingModel(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

    def forward(self, input_ids, attention_masks, global_attention_mask, labels):
        outputs = self.model(input_ids, attention_masks, global_attention_mask, labels=labels, return_dict=True)

        return outputs.loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        
        # TODO: Globally attend to query tokens
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)

        loss = self(input_ids, attention_masks, global_attention_mask, labels)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.args.learning_rate)

    def train_dataloader(self):
        logging.info("Loading feature file")
        feature_file = self._feature_file("train")

        features = torch.load(feature_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_labels),
            batch_size=self.args.train_batch_size,
            shuffle= True
        )

    def _feature_file(self, mode):
        cached_file_name = f"cached_{mode}"
        cached_file_path = os.path.join(self.args.data_dir, cached_file_name)
        
        return cached_file_path
    
    def _encode(self, query, body):
        return self.tokenizer(
            query, body, 
            max_length=self.args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )

    def prepare_data(self):
        feature_file_path = self._feature_file("train")

        if not args.overwrite_cache and os.path.exists(feature_file_path):
            logging.info("Using cached feature file")
            return

        logging.info("Creating features from dataset file at %s", args.data_dir)
        triples_path = os.path.join(self.args.data_dir, "triples.tsv")
        
        features = []

        with open(triples_path) as f:
            rows = csv.reader(f, delimiter="\t")

            for row in rows:
                triple = QueryTriple(*row)

                # Create a positive and negative feature
                inputs = self._encode(triple.query, triple.rel_doc_body)
                inputs["label"] = 1
                features.append(inputs)
                
                inputs = self._encode(triple.query, triple.rnd_doc_body)
                inputs["label"] = 0
                features.append(inputs)

        torch.save(features , feature_file_path)
        logging.info(f"Cached features to {feature_file_path}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Program arguments
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        # required=True,
        help="The data directory",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--overwrite-cache", type=bool, default=False, help="overwrite cache")

    # Training arguments
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)

    # Model arguments
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    wandb_logger = WandbLogger(project="adhoc-ranking")

    trainer = Trainer.from_argparse_args(
        args,
        logger=wandb_logger
    )

    model = RankingModel(args)

    trainer.fit(model)
    logging.info("Training complete")