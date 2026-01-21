import os
import argparse
import random
import csv
import json
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, load_from_disk
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.amp import autocast, GradScaler

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

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = True
        gold_mask = kwargs.pop("gold_mask", None)
        if gold_mask is None and hasattr(self, "gold_mask"):
            gold_mask = self.gold_mask
        outputs = super().forward(*args, **kwargs)
        attn_probs = outputs[1]
        if gold_mask is not None:
            gm = gold_mask.unsqueeze(1).unsqueeze(2)
            guided = attn_probs * (1 + self.lambda_guidance * gm)
            guided = guided / (guided.sum(dim=-1, keepdim=True) + 1e-9)
            outputs = (outputs[0], guided) + outputs[2:]
        return outputs

def align_explanation_to_tokens(example, tokenizer, max_len):
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    explanation = example.get("explanation_1", "") or ""
    text = premise + " [SEP] " + hypothesis
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    mask = [0] * len(input_ids)
    exp_lower = explanation.lower()
    text_lower = text.lower()
    for i, (start, end) in enumerate(offsets):
        if start == end: continue
        token_str = text_lower[start:end]
        if token_str.strip() and token_str in exp_lower:
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
    print(">> Downloading and preprocessing eSNLI...")
    dataset = load_dataset("esnli", cache_dir=args.cache_dir)
    dataset = dataset.map(lambda x: align_explanation_to_tokens(x, tokenizer, args.max_len))
    dataset.save_to_disk(cache_path)
    return dataset

def collate_fn(batch):
    return {k: torch.tensor([b[k] for b in batch]) for k in ["input_ids", "attention_mask", "gold_mask", "label"]}

def attention_iou(attn, gold_mask, k=10):
    scores = attn.mean(dim=0).mean(dim=0)
    actual_k = min(k, scores.size(0))
    topk = scores.topk(actual_k).indices
    pred_mask = torch.zeros_like(scores)
    pred_mask[topk] = 1
    intersection = (pred_mask * gold_mask).sum()
    union = (pred_mask + gold_mask).clamp(0, 1).sum()
    return (intersection / (union + 1e-9)).item()

def save_detailed_checkpoint(model, tokenizer, args, stats, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    args_dict = vars(args)
    with open(os.path.join(save_path, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    with open(os.path.join(save_path, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f">> [Checkpoint] Saved to: {save_path}")

def evaluate(model, loader, device):
    model.eval()
    correct, total, ious = 0, 0, []
    with torch.no_grad():
        for batch in loader:
            input_ids, labels = batch["input_ids"].to(device), batch["label"].to(device)
            attention_mask, gold_mask = batch["attention_mask"].to(device), batch["gold_mask"].to(device)
            for layer in model.bert.encoder.layer[-4:]:
                if isinstance(layer.attention.self, GuidedBertSelfAttention):
                    layer.attention.self.gold_mask = gold_mask
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
            correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            last_attn = outputs.attentions[-1]
            for i in range(input_ids.size(0)):
                ious.append(attention_iou(last_attn[i].cpu(), gold_mask[i].cpu()))
    return correct / total, sum(ious) / len(ious) if ious else 0.0

def train_one_epoch(model, loader, optimizer, device, mode, scaler, writer, epoch):
    model.train()
    total_loss = 0
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        input_ids, labels = batch["input_ids"].to(device), batch["label"].to(device)
        attention_mask, gold_mask = batch["attention_mask"].to(device), batch["gold_mask"].to(device)
        if mode == "guided":
            for layer in model.bert.encoder.layer[-4:]:
                if isinstance(layer.attention.self, GuidedBertSelfAttention):
                    layer.attention.self.gold_mask = gold_mask
        with autocast(device_type=device_type):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if step % 100 == 0:
            writer.add_scalar("Loss/step", loss.item(), epoch * len(loader) + step)
    return total_loss / len(loader)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mode_{args.mode}_lambda{args.lambda_guidance}_{timestamp}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_or_prepare_dataset(tokenizer, args)
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=collate_fn)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, attn_implementation="eager")
    if args.mode == "guided":
        for layer in model.bert.encoder.layer[-4:]:
            new_attn = GuidedBertSelfAttention(model.config, args.lambda_guidance)
            new_attn.load_state_dict(layer.attention.self.state_dict())
            layer.attention.self = new_attn
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(device=device.type if device.type != 'cpu' else 'cuda')
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.mode, scaler, writer, epoch)
        val_acc, val_iou = evaluate(model, val_loader, device)
        current_stats = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_iou": round(val_iou, 4),
            "mode": args.mode,
            "lambda": args.lambda_guidance
        }
        print(f"\n>> {current_stats}")
        writer.add_scalar("Loss/epoch", train_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)
        epoch_path = os.path.join(run_dir, f"checkpoint_ep{epoch + 1}")
        save_detailed_checkpoint(model, tokenizer, args, current_stats, epoch_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(run_dir, "best_model")
            save_detailed_checkpoint(model, tokenizer, args, current_stats, best_path)
            print(f"*** New Best Acc: {best_acc:.4f} saved! ***")
    file_exists = os.path.isfile(args.results_file)
    with open(args.results_file, 'a', newline='', encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=current_stats.keys())
        if not file_exists:
            writer_csv.writeheader()
        writer_csv.writerow(current_stats)
    writer.close()
    print(f">> Training complete. Run dir: {run_dir}")

if __name__ == "__main__":
    main()
