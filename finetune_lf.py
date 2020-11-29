import argparse
import logging
import csv
import os
import ctypes as ct
from collections import namedtuple

from tqdm import tqdm

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

# Set the max field size in csv, due to long text fields
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

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

        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        
        # TODO: Globally attend to query tokens
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=self.device)

        loss, logits = self(input_ids, attention_masks, global_attention_mask, labels)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch

        # TODO: Globally attend to query tokens
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=self.device)

        loss, logits = self(input_ids, attention_masks, global_attention_mask, labels)

        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss)
        self.log("acc", accuracy(preds, labels))

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                        num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.max_steps)

        return [optimizer], [scheduler]

    def get_dataloader(self, mode):
        logging.info(f"Loading {mode} feature file")
        feature_file = self._feature_file(mode)

        features = torch.load(feature_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_labels),
            batch_size=self.args.train_batch_size,
            shuffle=(mode == "train"),
            num_workers=4
        )

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

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
        for mode in ["train", "dev"]:
            feature_file_path = self._feature_file(mode)

            if not args.overwrite_cache and os.path.exists(feature_file_path):
                logging.info(f"Using cached {mode} feature file")
                continue

            logging.info(f"Creating {mode} features from dataset file at %s", args.data_dir)
            triples_path = os.path.join(self.args.data_dir, f"{mode}_triples.tsv")

            features = []

            with open(triples_path) as f:
                rows = csv.reader(f, delimiter="\t")

                for row in tqdm(rows):
                    # Bad row
                    if len(row) != 10:
                        continue

                    triple = QueryTriple(*row)

                    # Create a positive and negative feature
                    inputs = self._encode(triple.query, triple.rel_doc_body)
                    inputs["label"] = 1
                    features.append(inputs)

                    inputs = self._encode(triple.query, triple.rnd_doc_body)
                    inputs["label"] = 0
                    features.append(inputs)

            logging.info(f"{mode} dataset size: {len(features)}")

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
        default_root_dir=args.output_dir,
        logger=wandb_logger
    )

    model = RankingModel(args)

    trainer.fit(model)
    logging.info("Training complete")