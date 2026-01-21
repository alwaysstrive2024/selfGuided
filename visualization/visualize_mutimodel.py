import os
import torch
import torch.nn.functional as F
import numpy as np
import html
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk

# ====================== 1. 核心配置 ======================
# 定义你要对比的模型列表
MODELS_CONFIG = [
    {"name": "Guided Model", "id": "mode_guided_lambda1.0_20260120_184228"},
    {"name": "Vanilla Model", "id": "mode_vanilla_lambda1.0_20260120_215539"},
    {"name": "Random Model", "id": "mode_random_lambda1.0_20260121_003525"},
]

CHECKPOINTS_DIR = "../checkpoints"
DATA_CACHE = "./data_cache/esnli_tokenized"
VIS_DIR = "visualization/visualization_comparison"
os.makedirs(VIS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(VIS_DIR, "multi_model_comparison.html")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 100
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


# ====================== 3. 辅助函数 ======================
def get_color_style(score, color_type='blue'):
    score = max(0.0, min(1.0, score))
    if score < 0.05: return "background-color: transparent; color: black;"
    if color_type == 'red':  # Human
        r, g, b = 255, int(255 * (1 - score)), int(255 * (1 - score))
    elif color_type == 'blue':  # Guided
        r, g, b = int(255 * (1 - score)), int(255 * (1 - score * 0.6)), 255
    elif color_type == 'green':  # Vanilla
        r, g, b = int(255 * (1 - score)), 255, int(255 * (1 - score))
    else:  # Random (Purple-ish)
        r, g, b = int(255 * (1 - score * 0.5)), int(255 * (1 - score)), 255

    text_color = "white" if score > 0.5 else "black"
    return f"background-color: rgb({r}, {g}, {b}); color: {text_color};"


# ====================== 4. 主程序 ======================

def main():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_from_disk(DATA_CACHE)
    val_dataset = dataset["validation"]

    # --- A. 预加载所有模型 ---
    models = []
    for cfg in MODELS_CONFIG:
        path = os.path.join(CHECKPOINTS_DIR, cfg['id'], "best_model")
        print(f">> Loading {cfg['name']}...")
        m = BertForSequenceClassification.from_pretrained(path, output_attentions=True).to(DEVICE)
        m.eval()
        models.append({"name": cfg['name'], "model": m, "id": cfg['id']})

    all_samples_html = ""

    # --- B. 循环处理样本 ---
    for i in range(NUM_SAMPLES):
        example = val_dataset[i]
        inputs = {k: torch.tensor(v).unsqueeze(0).to(DEVICE) for k, v in example.items() if
                  k in ['input_ids', 'attention_mask']}

        tokens_raw = tokenizer.convert_ids_to_tokens(example["input_ids"])
        valid_indices = [idx for idx, t in enumerate(tokens_raw) if t != '[PAD]']
        f_tokens = [tokens_raw[idx] for idx in valid_indices]
        f_gold = np.array(example["gold_mask"])[valid_indices]
        true_label = LABEL_MAP.get(example["label"])

        # 生成各模型的比对块
        model_comparison_html = ""
        for idx, m_obj in enumerate(models):
            with torch.no_grad():
                out = m_obj['model'](**inputs)
                probs = F.softmax(out.logits, dim=-1)
                conf, pred_idx = torch.max(probs, dim=-1)
                attn = out.attentions[-1].mean(dim=1).squeeze(0).mean(dim=0)

            f_attn = attn.cpu().numpy()[valid_indices]
            norm_attn = (f_attn - f_attn.min()) / (f_attn.max() - f_attn.min() + 1e-9)

            pred_label = LABEL_MAP.get(pred_idx.item())
            is_correct = (example["label"] == pred_idx.item())
            res_color = "#28a745" if is_correct else "#dc3545"

            # 选择不同的颜色主题
            theme = ['blue', 'green', 'purple'][idx % 3]

            tokens_html = "".join([
                                      f'<span class="token" style="{get_color_style(s, theme)}" title="Score: {raw_s:.4f}">{html.escape(t)}</span>'
                                      for t, raw_s, s in zip(f_tokens, f_attn, norm_attn)])

            model_comparison_html += f"""
            <div class="model-row">
                <div class="model-meta">
                    <span class="model-tag">{m_obj['name']}</span>
                    <span style="color: {res_color}; font-weight: bold;">{pred_label} ({conf.item():.2%})</span>
                </div>
                <div class="text-box">{tokens_html}</div>
            </div>
            """

        # 组装单个样本卡片
        gold_tokens_html = "".join([f'<span class="token" style="{get_color_style(s, "red")}">{html.escape(t)}</span>'
                                    for t, s in zip(f_tokens, f_gold)])

        all_samples_html += f"""
        <div class="sample-card">
            <div class="card-header">Sample #{i} | <small>Ground Truth: <b>{true_label}</b></small></div>
            <div class="explanation-box"><b>Human Rationale:</b> "{html.escape(example.get('explanation_1', 'N/A'))}"</div>

            <h3>Model Comparisons</h3>
            {model_comparison_html}

            <h3>Ground Truth Mask</h3>
            <div class="text-box">{gold_tokens_html}</div>
        </div>
        """

    # --- C. 最终 HTML 包装 ---
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>e-SNLI Cross-Model Comparison</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 40px; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .sample-card {{ background: white; border-radius: 12px; padding: 25px; margin-bottom: 50px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
            .card-header {{ font-size: 20px; font-weight: bold; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }}
            .explanation-box {{ background: #fffbe6; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-style: italic; border: 1px solid #ffe58f; }}

            .model-row {{ display: flex; align-items: flex-start; margin-bottom: 15px; border-bottom: 1px solid #f0f0f0; padding-bottom: 10px; }}
            .model-meta {{ width: 220px; flex-shrink: 0; padding-right: 20px; border-right: 1px solid #eee; }}
            .model-tag {{ display: block; font-size: 12px; color: #777; text-transform: uppercase; font-weight: bold; margin-bottom: 5px; }}

            .text-box {{ flex-grow: 1; padding-left: 20px; line-height: 2.2; font-size: 16px; }}
            .token {{ display: inline-block; padding: 0 4px; margin: 1px; border-radius: 3px; border: 1px solid #fdfdfd; }}
            h3 {{ font-size: 14px; color: #999; text-transform: uppercase; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>e-SNLI Cross-Model Explainability Analysis</h1>
            <p>Comparing Attention Heatmaps across different training strategies.</p>
            {all_samples_html}
        </div>
    </body>
    </html>
    """

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\n✅ 成功生成多模型对比报告：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()