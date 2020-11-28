import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

from transformers import (
    AdamW,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizerFast,
    get_linear_schedule_with_warmup
)

from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

class RankingModel(LightningModule):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.linear(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)

        loss = F.cross_entropy(preds, labels)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.params.learning_rate)

    def train_dataloader(self):
        transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))
                                ])

        dataset = datasets.MNIST(root=self.params.data_dir, train=True, transform=transform, download=True)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.params.train_batch_size,
            shuffle=True,
            num_workers=4
        )

        return loader

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Program arguments
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The data directory",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

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