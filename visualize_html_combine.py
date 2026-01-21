import os
import torch
import torch.nn.functional as F
import numpy as np
import html
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk

# ====================== 1. 配置参数 ======================
RUN_ID = "mode_random_lambda1.0_20260121_003525"
CHECKPOINTS_DIR = "./checkpoints"
RUN_DIR = os.path.join(CHECKPOINTS_DIR, RUN_ID)
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model")
DATA_CACHE = "./data_cache/esnli_tokenized"

# 输出文件名
VIS_DIR = os.path.join("./visualization_combined", RUN_ID)
os.makedirs(VIS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(VIS_DIR, "combined_report.html")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 20  # 设置你想一起对比的样本数量

LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


# ====================== 2. 模型结构定义 ======================
class GuidedBertSelfAttention(BertSelfAttention):
    def __init__(self, config, lambda_guidance=1.0):
        super().__init__(config)
        self.lambda_guidance = lambda_guidance

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = True
        gold_mask = kwargs.pop("gold_mask", None)
        outputs = super().forward(*args, **kwargs)
        attn_probs = outputs[1]
        if gold_mask is not None:
            gm = gold_mask.unsqueeze(1).unsqueeze(2)
            guided = attn_probs * (1 + self.lambda_guidance * gm)
            guided = guided / (guided.sum(dim=-1, keepdim=True) + 1e-9)
            outputs = (outputs[0], guided) + outputs[2:]
        return outputs


# ====================== 3. HTML 渲染工具 ======================

def get_color_style(score, is_gold=False):
    score = max(0.0, min(1.0, score))
    if score < 0.05: return "background-color: transparent; color: black;"
    if is_gold:
        r, g, b = 255, int(255 * (1 - score)), int(255 * (1 - score))
    else:
        r, g, b = int(255 * (1 - score)), int(255 * (1 - score * 0.6)), 255
    text_color = "white" if score > 0.5 else "black"
    return f"background-color: rgb({r}, {g}, {b}); color: {text_color};"


def generate_sample_html(data):
    """为单个样本生成 HTML 代码块"""
    is_correct = data['true_label'] == data['pred_label']
    status_class = "correct-card" if is_correct else "incorrect-card"
    status_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"

    # 构造 Token 序列 HTML
    attn_tokens_html = ""
    for t, raw_s, norm_s in zip(data['tokens'], data['attn_scores'], data['norm_attn']):
        style = get_color_style(norm_s, is_gold=False)
        attn_tokens_html += f'<span class="token" style="{style}" title="Score: {raw_s:.4f}">{html.escape(t)}</span>'

    gold_tokens_html = ""
    for t, gold_s in zip(data['tokens'], data['gold_mask']):
        style = get_color_style(gold_s, is_gold=True)
        gold_tokens_html += f'<span class="token" style="{style}">{html.escape(t)}</span>'

    return f"""
    <div class="sample-anchor" id="sample-{data['id']}"></div>
    <div class="sample-card {status_class}">
        <div class="card-header">
            <span class="sample-id">Sample #{data['id']}</span>
            <span class="status-badge">{status_text}</span>
        </div>

        <div class="meta-section">
            <div class="meta-item"><b>Ground Truth:</b> {data['true_label']}</div>
            <div class="meta-item"><b>Prediction:</b> {data['pred_label']}</div>
            <div class="meta-item"><b>Confidence:</b> {data['confidence']:.2%}</div>
        </div>

        <div class="explanation-section">
            <strong>Human Interpretation:</strong> "{html.escape(data['explanation'])}"
        </div>

        <div class="viz-section">
            <div class="viz-title">🔍 Model Attention Heatmap</div>
            <div class="text-box">{attn_tokens_html}</div>
        </div>

        <div class="viz-section">
            <div class="viz-title">🎯 Human Gold Rationales</div>
            <div class="text-box">{gold_tokens_html}</div>
        </div>
    </div>
    """


# ====================== 4. 主程序 ======================

def main():
    print(">> Initializing model...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        BEST_MODEL_PATH, output_attentions=True, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.eval()

    dataset = load_from_disk(DATA_CACHE)
    val_dataset = dataset["validation"]

    samples_data = []
    correct_count = 0

    print(f">> Processing {NUM_SAMPLES} samples...")
    for i in range(NUM_SAMPLES):
        example = val_dataset[i]

        # 模型推理
        inputs = {k: torch.tensor(v).unsqueeze(0).to(DEVICE) for k, v in example.items() if
                  k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)
            attn = outputs.attentions[-1].mean(dim=1).squeeze(0).mean(dim=0)

        # 准备数据
        tokens_raw = tokenizer.convert_ids_to_tokens(example["input_ids"])
        valid_indices = [idx for idx, t in enumerate(tokens_raw) if t != '[PAD]']

        f_tokens = [tokens_raw[idx] for idx in valid_indices]
        f_attn = attn.cpu().numpy()[valid_indices]
        f_gold = np.array(example["gold_mask"])[valid_indices]
        norm_attn = (f_attn - f_attn.min()) / (f_attn.max() - f_attn.min() + 1e-9)

        is_correct = (example["label"] == pred_idx.item())
        if is_correct: correct_count += 1

        samples_data.append({
            "id": i,
            "tokens": f_tokens,
            "attn_scores": f_attn,
            "norm_attn": norm_attn,
            "gold_mask": f_gold,
            "true_label": LABEL_MAP.get(example["label"], str(example["label"])),
            "pred_label": LABEL_MAP.get(pred_idx.item(), str(pred_idx.item())),
            "confidence": conf.item(),
            "explanation": example.get("explanation_1", example.get("explanation", "N/A")),
            "is_correct": is_correct
        })

    # ====================== 5. 组装最终 HTML ======================

    # 导航锚点
    nav_html = "".join(
        [f'<a href="#sample-{d["id"]}" class="nav-dot {"dot-err" if not d["is_correct"] else ""}">{d["id"]}</a>' for d
         in samples_data])

    # 所有样本内容
    samples_html = "".join([generate_sample_html(d) for d in samples_data])

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>e-SNLI Comparison Report</title>
        <style>
            body {{ font-family: 'Inter', -apple-system, sans-serif; background-color: #f4f7f6; margin: 0; padding: 0; }}
            .header-bar {{ background: #2c3e50; color: white; padding: 20px 40px; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 10px rgba(0,0,0,0.2); display: flex; justify-content: space-between; align-items: center; }}
            .stats {{ font-size: 18px; }}
            .nav-container {{ padding: 10px 40px; background: #fff; border-bottom: 1px solid #ddd; position: sticky; top: 70px; z-index: 99; overflow-x: auto; white-space: nowrap; }}
            .nav-dot {{ display: inline-block; width: 30px; height: 30px; line-height: 30px; text-align: center; margin: 2px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; font-size: 12px; font-weight: bold; }}
            .nav-dot.dot-err {{ background: #dc3545; }}

            .content {{ max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
            .sample-card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 50px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); border-left: 10px solid #ddd; position: relative; }}
            .sample-card.correct-card {{ border-left-color: #28a745; }}
            .sample-card.incorrect-card {{ border-left-color: #dc3545; }}

            .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
            .sample-id {{ font-size: 24px; font-weight: bold; color: #333; }}
            .status-badge {{ padding: 5px 12px; border-radius: 6px; color: white; font-weight: bold; font-size: 14px; }}
            .correct-card .status-badge {{ background: #28a745; }}
            .incorrect-card .status-badge {{ background: #dc3545; }}

            .meta-section {{ display: flex; gap: 40px; margin-bottom: 20px; background: #f8f9fa; padding: 15px; border-radius: 8px; }}
            .explanation-section {{ margin-bottom: 25px; padding: 15px; background: #fffdf0; border: 1px dashed #ffe58f; border-radius: 8px; font-style: italic; color: #856404; }}

            .viz-section {{ margin-bottom: 20px; }}
            .viz-title {{ font-size: 14px; color: #999; text-transform: uppercase; font-weight: bold; margin-bottom: 10px; }}
            .text-box {{ line-height: 2.5; font-size: 18px; }}
            .token {{ display: inline-block; padding: 0 6px; margin: 2px; border-radius: 4px; border: 1px solid #eee; cursor: help; }}
            .sample-anchor {{ position: relative; top: -130px; visibility: hidden; }}
        </style>
    </head>
    <body>
        <div class="header-bar">
            <h1 style="margin:0;">e-SNLI Explainability Report</h1>
            <div class="stats">Accuracy: <b>{correct_count / NUM_SAMPLES:.1%}</b> ({correct_count}/{NUM_SAMPLES})</div>
        </div>

        <div class="nav-container">
            <strong>Jump to:</strong> {nav_html}
        </div>

        <div class="content">
            {samples_html}
        </div>

        <div style="text-align: center; color: #aaa; padding: 40px;">
            End of Report - Run ID: {RUN_ID}
        </div>
    </body>
    </html>
    """

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"\n✨ 全景报告已生成！请查看：\n{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()