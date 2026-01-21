import os
import torch
import torch.nn.functional as F
import numpy as np
import html
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk

# ====================== 1. 配置参数 ======================
RUN_ID = "mode_random_lambda1.0_20260121_003525"  # 替换为你的实际文件夹名
CHECKPOINTS_DIR = "./checkpoints"
RUN_DIR = os.path.join(CHECKPOINTS_DIR, RUN_ID)
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model")
DATA_CACHE = "./data_cache/esnli_tokenized"

# 输出目录
VIS_DIR = os.path.join("./visualization_detailed", RUN_ID)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 15  # 想要生成的样本数量

# e-SNLI 标签映射
LABEL_MAP = {0: "Entailment (蕴含)", 1: "Neutral (中性)", 2: "Contradiction (矛盾)"}


# ====================== 2. 模型结构定义 ======================
# 必须保留此定义以确保能够正确加载包含自定义层权重的模型
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


# ====================== 3. HTML 渲染核心逻辑 ======================

def get_color_style(score, is_gold=False):
    """根据分数生成 CSS 颜色样式"""
    score = max(0.0, min(1.0, score))
    if score < 0.05: return "background-color: transparent; color: black;"

    if is_gold:
        # 红色调 (Human)
        r, g, b = 255, int(255 * (1 - score)), int(255 * (1 - score))
    else:
        # 蓝色调 (Model)
        r, g, b = int(255 * (1 - score)), int(255 * (1 - score * 0.6)), 255

    text_color = "white" if score > 0.5 else "black"
    return f"background-color: rgb({r}, {g}, {b}); color: {text_color};"


def save_detailed_html(sample_id, data, output_dir):
    """生成单个样本的详细 HTML 页面"""
    filename = f"sample_{sample_id}.html"
    filepath = os.path.join(output_dir, filename)

    is_correct = data['true_label'] == data['pred_label']
    status_color = "#28a745" if is_correct else "#dc3545"
    status_text = "CORRECT" if is_correct else "INCORRECT"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 30px; background-color: #f0f2f5; color: #333; line-height: 1.6; }}
            .container {{ max-width: 1000px; margin: auto; }}
            .meta-card {{ 
                background: white; padding: 25px; border-radius: 15px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 25px;
                border-top: 10px solid {status_color};
            }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 15px; }}
            .info-label {{ font-size: 13px; color: #888; text-transform: uppercase; font-weight: bold; }}
            .info-value {{ font-size: 18px; font-weight: 600; color: #1a1a1a; display: block; margin-top: 5px; }}

            .visual-card {{ background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 25px; }}
            h3 {{ margin: 0 0 15px 0; font-size: 15px; color: #666; text-transform: uppercase; border-bottom: 1px solid #eee; padding-bottom: 8px; }}

            .text-box {{ line-height: 2.5; font-size: 18px; }}
            .token {{ display: inline-block; padding: 0px 6px; margin: 2px; border-radius: 4px; border: 1px solid #eee; transition: 0.2s; cursor: help; }}
            .token:hover {{ transform: scale(1.1); z-index: 10; }}

            .explanation-box {{ background: #fffbe6; padding: 20px; border-radius: 10px; border: 1px solid #ffe58f; color: #856404; font-size: 16px; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; color: white; font-weight: bold; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="meta-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h1 style="margin:0;">Sample #{sample_id}</h1>
                    <span class="badge" style="background-color: {status_color};">{status_text}</span>
                </div>
                <div class="grid">
                    <div><span class="info-label">Ground Truth</span><span class="info-value">{data['true_label']}</span></div>
                    <div><span class="info-label">Model Prediction</span><span class="info-value">{data['pred_label']}</span></div>
                    <div><span class="info-label">Confidence</span><span class="info-value">{data['confidence']:.2%}</span></div>
                </div>
            </div>

            <div class="visual-card">
                <h3>📝 Human Rationale (Text Explanation)</h3>
                <div class="explanation-box">
                    <strong>Interpretation:</strong> "{html.escape(data['explanation'])}"
                </div>
            </div>

            <div class="visual-card">
                <h3>🔍 Model Self-Attention Heatmap</h3>
                <div class="text-box">
    """
    # 渲染模型 Attention
    for t, raw_s, norm_s in zip(data['tokens'], data['attn_scores'], data['norm_attn']):
        style = get_color_style(norm_s, is_gold=False)
        html_content += f'<span class="token" style="{style}" title="Score: {raw_s:.4f}">{html.escape(t)}</span>\n'

    html_content += """
                </div>
            </div>

            <div class="visual-card">
                <h3>🎯 Ground Truth Rationales (Highlighted by Human)</h3>
                <div class="text-box">
    """
    # 渲染 Gold Mask
    for t, gold_s in zip(data['tokens'], data['gold_mask']):
        style = get_color_style(gold_s, is_gold=True)
        html_content += f'<span class="token" style="{style}">{html.escape(t)}</span>\n'

    html_content += "</div></div></div></body></html>"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filename


def create_index_html(results, output_dir):
    """创建左侧列表、右侧内容的预览页面"""
    index_path = os.path.join(output_dir, "index.html")
    list_items = ""
    for r in results:
        color = "#28a745" if r['correct'] else "#dc3545"
        list_items += f"""
        <a href="{r['file']}" target="viewer" class="nav-item">
            <span style="color: {color}">●</span> Sample {r['id']} 
            <small style="display:block; color:#999;">{r['pred_label']}</small>
        </a>"""

    html_code = f"""
    <html>
    <head>
        <title>e-SNLI Explainability Dashboard</title>
        <style>
            body {{ display: flex; height: 100vh; margin: 0; font-family: sans-serif; background: #2c3e50; }}
            #sidebar {{ width: 260px; background: #fff; overflow-y: auto; border-right: 1px solid #ddd; padding: 15px; }}
            #content {{ flex: 1; background: #f0f2f5; }}
            iframe {{ width: 100%; height: 100%; border: none; }}
            .nav-item {{ 
                display: block; padding: 12px; margin-bottom: 8px; 
                background: #f8f9fa; border-radius: 8px; text-decoration: none; color: #333;
                font-size: 14px; border: 1px solid transparent; transition: 0.2s;
            }}
            .nav-item:hover {{ background: #e9ecef; border-color: #dee2e6; }}
            h2 {{ font-size: 18px; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>Samples</h2>
            {list_items}
        </div>
        <div id="content">
            <iframe name="viewer" src="{results[0]['file'] if results else ''}"></iframe>
        </div>
    </body>
    </html>
    """
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_code)


# ====================== 4. 主程序 ======================

def main():
    print(f">> Loading model and tokenizer from {BEST_MODEL_PATH}...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # 自动处理自定义 Attention 类加载问题
    model = BertForSequenceClassification.from_pretrained(
        BEST_MODEL_PATH,
        output_attentions=True,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.eval()

    dataset = load_from_disk(DATA_CACHE)
    val_dataset = dataset["validation"]

    generated_info = []

    for i in range(NUM_SAMPLES):
        example = val_dataset[i]

        # 推理
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(DEVICE)
        mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=mask)
            # 获取预测
            probs = F.softmax(outputs.logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)
            # 获取最后一层 Attention 并平均所有 Head
            attn = outputs.attentions[-1].mean(dim=1).squeeze(0).mean(dim=0)

        # 整理数据
        tokens_raw = tokenizer.convert_ids_to_tokens(example["input_ids"])
        valid_indices = [idx for idx, t in enumerate(tokens_raw) if t != '[PAD]']

        f_tokens = [tokens_raw[idx] for idx in valid_indices]
        f_attn = attn.cpu().numpy()[valid_indices]
        f_gold = np.array(example["gold_mask"])[valid_indices]

        # 注意力归一化以便着色
        norm_attn = (f_attn - f_attn.min()) / (f_attn.max() - f_attn.min() + 1e-9)

        # 提取解释文本（检查 e-SNLI 的常见字段名）
        explanation = example.get("explanation_1", "N/A")
        if explanation == "N/A":  # 兼容某些缓存版本
            explanation = example.get("explanation", "No textual explanation found.")

        data_payload = {
            "tokens": f_tokens,
            "attn_scores": f_attn,
            "norm_attn": norm_attn,
            "gold_mask": f_gold,
            "true_label": LABEL_MAP.get(example["label"], str(example["label"])),
            "pred_label": LABEL_MAP.get(pred_idx.item(), str(pred_idx.item())),
            "confidence": conf.item(),
            "explanation": explanation
        }

        # 生成页面
        fname = save_detailed_html(i, data_payload, VIS_DIR)

        generated_info.append({
            "id": i,
            "file": fname,
            "correct": example["label"] == pred_idx.item(),
            "pred_label": data_payload["pred_label"]
        })
        print(f"[Sample {i}] Predicted: {data_payload['pred_label']} | Correct: {example['label'] == pred_idx.item()}")

    # 创建仪表盘索引
    create_index_html(generated_info, VIS_DIR)

    print(f"\n✨ 可视化已完成！请用浏览器打开以下路径查看结果：")
    print(f"{os.path.abspath(os.path.join(VIS_DIR, 'index.html'))}")


if __name__ == "__main__":
    main()