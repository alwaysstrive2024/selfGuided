import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_from_disk

# ---------------------- 配置 ----------------------
RUN_DIR = "./checkpoints/mode_guided_lambda1.0_20260120_184228"
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model")  # 可切换到 checkpoint_ep1 等
DATA_CACHE = "./data_cache/esnli_tokenized"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10
NUM_SAMPLES = 3  # 可视化前几条验证集样本
VIS_DIR = "./visualization/guided"
os.makedirs(VIS_DIR, exist_ok=True)

# ---------------------- 加载模型 & tokenizer ----------------------
tokenizer = BertTokenizerFast.from_pretrained(BEST_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(BEST_MODEL_PATH, output_attentions=True)
model.to(DEVICE)
model.eval()

# ---------------------- 加载验证集 ----------------------
dataset = load_from_disk(DATA_CACHE)
val_dataset = dataset["validation"]

# ---------------------- 工具函数 ----------------------
def collate_fn(batch):
    return {k: torch.tensor([b[k] for b in batch]) for k in ["input_ids", "attention_mask", "gold_mask", "label"]}

def save_attention_heatmap(input_ids, attn_probs, gold_mask, top_k=10, sample_idx=0, vis_dir=VIS_DIR):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attn_scores = attn_probs.cpu().numpy()
    gold_mask_np = gold_mask.cpu().numpy()

    # Attention 热力图
    plt.figure(figsize=(min(len(tokens)*0.3, 15), 2))
    sns.heatmap([attn_scores], annot=[tokens], fmt='', cmap="Blues", cbar=True)
    plt.title("Token Attention Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"sample_{sample_idx}_attention.png"))
    plt.close()

    # Gold mask 热力图
    plt.figure(figsize=(min(len(tokens)*0.3, 2), 2))
    sns.heatmap([gold_mask_np], annot=[tokens], fmt='', cmap="Reds", cbar=True)
    plt.title("Gold Mask")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"sample_{sample_idx}_gold_mask.png"))
    plt.close()

    # Top-K token
    topk_idx = attn_scores.argsort()[-top_k:][::-1]
    topk_tokens = [tokens[i] for i in topk_idx]
    topk_scores = attn_scores[topk_idx]
    plt.figure(figsize=(min(len(tokens)*0.3, 12), 4))
    sns.barplot(x=topk_tokens, y=topk_scores)
    plt.title(f"Top-{top_k} Attention Tokens")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"sample_{sample_idx}_topk.png"))
    plt.close()

# ---------------------- 绘制训练曲线 ----------------------
def plot_training_curves(run_dir, vis_dir=VIS_DIR):
    stats_files = []
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f == "stats.json":
                stats_files.append(os.path.join(root, f))

    if not stats_files:
        print("No stats.json files found for plotting.")
        return

    stats_list = []
    for f in sorted(stats_files):
        with open(f, "r", encoding="utf-8") as jf:
            stats_list.append(json.load(jf))

    epochs = [s["epoch"] for s in stats_list]
    train_loss = [s["train_loss"] for s in stats_list]
    val_acc = [s["val_accuracy"] for s in stats_list]
    val_iou = [s["val_iou"] for s in stats_list]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc, marker='o', label="Validation Accuracy")
    plt.plot(epochs, val_iou, marker='o', label="Validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Accuracy & IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "acc_iou_curve.png"))
    plt.close()

# ---------------------- 可视化样本 attention ----------------------
def visualize_samples(model, val_dataset, num_samples=3, top_k=10, vis_dir=VIS_DIR):
    print(f">> Saving visualizations for first {num_samples} validation samples...")
    for i in range(num_samples):
        example = val_dataset[i]
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(DEVICE)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(DEVICE)
        gold_mask = torch.tensor(example["gold_mask"]).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if outputs.attentions is not None:
                last_layer_attn = outputs.attentions[-1].mean(dim=1).squeeze(0).mean(dim=0)
            else:
                last_layer_attn = torch.zeros(len(example["input_ids"]))

        save_attention_heatmap(example["input_ids"], last_layer_attn, gold_mask, top_k=top_k, sample_idx=i, vis_dir=vis_dir)

# ---------------------- 主函数 ----------------------
def main():
    print(">> Saving training curves...")
    plot_training_curves(RUN_DIR, VIS_DIR)

    print(">> Saving token attention visualizations...")
    visualize_samples(model, val_dataset, NUM_SAMPLES, TOP_K, VIS_DIR)

    print(f">> All visualizations saved in {VIS_DIR}")

if __name__ == "__main__":
    main()
