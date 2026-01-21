import os
import argparse
import random
import csv
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, load_from_disk
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)
from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertSelfAttention


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="guided",
                        choices=["guided", "vanilla", "random"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lambda_guidance", type=float, default=1.0)
    parser.add_argument("--cache_dir", type=str, default="./data_cache")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_file", type=str, default="results.csv")
    return parser.parse_args()



class GuidedBertSelfAttention(BertSelfAttention):
    def __init__(self, config, lambda_guidance=1.0):
        super().__init__(config)
        self.lambda_guidance = lambda_guidance

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        gold_mask=None,
    ):
        outputs = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions=True
        )

        attn_probs = outputs[1]  # (batch, heads, seq, seq)

        if gold_mask is not None:
            gm = gold_mask.unsqueeze(1).unsqueeze(2)  # (batch,1,1,seq)
            guided = attn_probs * (1 + self.lambda_guidance * gm)
            guided = guided / guided.sum(dim=-1, keepdim=True)
            outputs = (outputs[0], guided) + outputs[2:]

        return outputs



def align_explanation_to_tokens(example, tokenizer, max_len):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    explanation = example["explanation_1"]

    text = premise + " [SEP] " + hypothesis

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]

    mask = [0] * len(input_ids)

    exp = explanation.lower()
    text_lower = text.lower()

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        token_str = text_lower[start:end]
        if token_str.strip() and token_str in exp:
            mask[i] = 1

    encoding["gold_mask"] = mask
    encoding.pop("offset_mapping")
    encoding["label"] = example["label"]

    return encoding


def load_or_prepare_dataset(tokenizer, args):
    cache_path = os.path.join(args.cache_dir, "esnli_tokenized")

    if os.path.exists(cache_path):
        print(">> Loading tokenized dataset from disk...")
        return load_from_disk(cache_path)

    print(">> Downloading eSNLI...")
    # Use this Parquet-compatible version
    dataset = load_dataset("esnli", trust_remote_code=True, cache_dir=args.cache_dir)
    # dataset = load_dataset("lucasmccabe-lmi/esnli", cache_dir=args.cache_dir)

    print(">> Tokenizing and aligning explanations...")
    dataset = dataset.map(
        lambda x: align_explanation_to_tokens(x, tokenizer, args.max_len),
        batched=False
    )

    os.makedirs(args.cache_dir, exist_ok=True)
    dataset.save_to_disk(cache_path)
    print(">> Saved tokenized dataset to disk.")

    return dataset

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.tensor([b[k] for b in batch])
    return out

def randomize_gold_mask(batch):
    gm = batch["gold_mask"]
    for i in range(len(gm)):
        ones = gm[i].sum().item()
        idx = torch.randperm(len(gm[i]))[:ones]
        new = torch.zeros_like(gm[i])
        new[idx] = 1
        gm[i] = new
    batch["gold_mask"] = gm
    return batch

def attention_iou(attn, gold_mask, k=10):
    scores = attn.mean(dim=0).mean(dim=0)  # (seq,)
    topk = scores.topk(k).indices

    pred_mask = torch.zeros_like(scores)
    pred_mask[topk] = 1

    intersection = (pred_mask * gold_mask).sum()
    union = (pred_mask + gold_mask).clamp(0, 1).sum()

    if union.item() == 0:
        return 0.0

    return (intersection / union).item()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    ious = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            gold_mask = batch["gold_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gold_mask=gold_mask,
                output_attentions=True
            )

            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            attns = outputs.attentions
            last_attn = attns[-1][0]  # (heads, seq, seq)
            iou = attention_iou(last_attn.cpu(), gold_mask[0].cpu())
            ious.append(iou)

    acc = correct / total
    mean_iou = sum(ious) / len(ious)

    return acc, mean_iou

def train_one_epoch(model, loader, optimizer, device, mode):
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        gold_mask = batch["gold_mask"].to(device)

        if mode == "random":
            batch = randomize_gold_mask(batch)
            gold_mask = batch["gold_mask"].to(device)

        if mode == "vanilla":
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                gold_mask=gold_mask,
                output_attentions=True
            )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def save_results(args, final_loss, final_acc, final_iou):
    file_exists = os.path.exists(args.results_file)

    with open(args.results_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "mode",
                "epochs",
                "batch_size",
                "lr",
                "lambda_guidance",
                "final_loss",
                "final_acc",
                "final_iou"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            args.mode,
            args.epochs,
            args.batch_size,
            args.lr,
            args.lambda_guidance if args.mode == "guided" else "NA",
            round(final_loss, 4),
            round(final_acc, 4),
            round(final_iou, 4)
        ])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir=f"runs/esnli_{args.mode}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_or_prepare_dataset(tokenizer, args)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

    if args.mode == "guided":
        for layer in model.bert.encoder.layer[-4:]:
            layer.attention.self = GuidedBertSelfAttention(
                model.config, args.lambda_guidance
            )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    final_loss, final_acc, final_iou = None, None, None

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, args.mode
        )

        acc, mean_iou = evaluate(model, val_loader, device)

        final_loss, final_acc, final_iou = loss, acc, mean_iou

        print(
            f"[{args.mode}] Epoch {epoch+1} | "
            f"Loss: {loss:.4f} | Acc: {acc:.4f} | IoU: {mean_iou:.4f}"
        )

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/val", acc, epoch)
        writer.add_scalar("IoU/val", mean_iou, epoch)

        save_path = os.path.join(
            args.save_dir, f"bert_esnli_{args.mode}_epoch{epoch+1}"
        )
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    save_results(args, final_loss, final_acc, final_iou)

    writer.close()
    print(">> Training finished.")
    print(f">> Final results saved to {args.results_file}")


if __name__ == "__main__":
    main()
