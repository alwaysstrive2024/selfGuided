import os
import torch
import numpy as np
import html  # 用于转义特殊字符
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk

# ---------------------- 配置 ----------------------
RUN_ID = "mode_random_lambda1.0_20260121_003525"  # 请替换为你的实际文件夹名
CHECKPOINTS_DIR = "./checkpoints"
RUN_DIR = os.path.join(CHECKPOINTS_DIR, RUN_ID)
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model")
DATA_CACHE = "./data_cache/esnli_tokenized"

# HTML 输出目录
VIS_DIR = os.path.join("./visualization_html", RUN_ID)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 10  # 想要生成的样本数量


# ---------------------- 模型定义 (保持一致) ----------------------
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


# ---------------------- HTML 生成工具函数 ----------------------
def get_color_style(score, is_gold=False):
    """
    根据分数生成 CSS 背景色样式。
    Score 范围预计在 0.0 ~ 1.0 之间。
    """
    # 限制范围，防止越界
    score = max(0.0, min(1.0, score))

    # 颜色越深，透明度越高。如果分数太小，直接给白色背景，保持干净
    if score < 0.05:
        return "background-color: transparent; color: black;"

    # 计算颜色 (RGB)
    # 蓝色 (Model): r=255->0, g=255->100, b=255 (保持蓝色通道高)
    # 红色 (Gold):  r=255, g=255->0, b=255->0 (保持红色通道高)

    if is_gold:
        # 红色调：分数越高，背景越红
        r = 255
        g = int(255 * (1 - score))
        b = int(255 * (1 - score))
    else:
        # 蓝色调：分数越高，背景越蓝
        r = int(255 * (1 - score))
        g = int(255 * (1 - score * 0.5))  # 让它偏一点青色，比较好看
        b = 255

    # 简单的对比度调整：如果背景太深，文字变成白色
    text_color = "white" if score > 0.7 else "black"

    return f"background-color: rgb({r}, {g}, {b}); color: {text_color};"


def save_html_sample(sample_id, tokens, attn_scores, gold_mask, output_dir):
    """生成单个样本的 HTML 片段文件"""

    # 1. 归一化 Attention Score (Min-Max) 以增强视觉对比度
    # 如果不做归一化，Attention往往很稀疏，颜色会非常淡看不清
    min_s = attn_scores.min()
    max_s = attn_scores.max()
    if max_s - min_s > 1e-9:
        norm_attn = (attn_scores - min_s) / (max_s - min_s)
    else:
        norm_attn = attn_scores

    filename = f"sample_{sample_id}.html"
    filepath = os.path.join(output_dir, filename)

    # HTML 头部和样式
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Sample {sample_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f9f9f9; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; }}
            h2 {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            h3 {{ margin-top: 0; font-size: 16px; color: #555; }}
            .text-box {{ line-height: 2.2; font-size: 16px; }}
            .token {{ 
                display: inline-block; 
                padding: 2px 5px; 
                margin: 2px; 
                border-radius: 4px; 
                border: 1px solid #eee;
                cursor: default;
                position: relative;
            }}
            /* Tooltip 样式 */
            .token:hover::after {{
                content: attr(data-score);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: #333;
                color: #fff;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                white-space: nowrap;
                z-index: 10;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <h2>Sample ID: {sample_id}</h2>
    """

    # --- Section 1: Model Attention (Blue) ---
    html_content += """
        <div class="card">
            <h3>🔹 Model Attention Prediction (Blue)</h3>
            <div class="text-box">
    """
    for token, raw_score, norm_score in zip(tokens, attn_scores, norm_attn):
        style = get_color_style(norm_score, is_gold=False)
        safe_token = html.escape(token)  # 防止 <UNK> 等符号破坏 HTML
        # data-score 属性用于显示 Tooltip
        html_content += f'<span class="token" style="{style}" data-score="{raw_score:.4f}">{safe_token}</span>\n'
    html_content += "</div></div>"

    # --- Section 2: Gold Mask (Red) ---
    html_content += """
        <div class="card">
            <h3>🔸 Ground Truth / Gold Mask (Red)</h3>
            <div class="text-box">
    """
    for token, score in zip(tokens, gold_mask):
        style = get_color_style(score, is_gold=True)
        safe_token = html.escape(token)
        html_content += f'<span class="token" style="{style}" data-score="{score:.1f}">{safe_token}</span>\n'
    html_content += "</div></div>"

    html_content += "</body></html>"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filename


def create_index_html(file_list, output_dir):
    """创建一个索引页，方便跳转查看所有样本"""
    index_path = os.path.join(output_dir, "index.html")
    links = ""
    for f in file_list:
        links += f'<li><a href="{f}" target="content_frame">{f}</a></li>\n'

    html = f"""
    <html>
    <head><title>Visualization Dashboard</title>
    <style>
        body {{ display: flex; height: 100vh; margin: 0; font-family: sans-serif; }}
        #sidebar {{ width: 200px; background: #f0f0f0; padding: 20px; overflow-y: auto; border-right: 1px solid #ccc; }}
        #content {{ flex: 1; }}
        iframe {{ width: 100%; height: 100%; border: none; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ margin-bottom: 10px; }}
        a {{ text-decoration: none; color: #007bff; font-weight: bold; }}
        a:hover {{ text-decoration: underline; }}
    </style>
    </head>
    <body>
        <div id="sidebar">
            <h3>Sample List</h3>
            <ul>{links}</ul>
        </div>
        <div id="content">
            <iframe name="content_frame" src="{file_list[0] if file_list else ''}"></iframe>
        </div>
    </body>
    </html>
    """
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Index created at: {index_path}")


# ---------------------- 主逻辑 ----------------------
def main():
    print(f">> Loading model from {BEST_MODEL_PATH} ...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    try:
        model = BertForSequenceClassification.from_pretrained(
            BEST_MODEL_PATH,
            output_attentions=True,
            attn_implementation="eager"
        )
    except Exception as e:
        print("Warning: Standard loading failed, trying to ignore mismatched keys if custom class logic is involved...")
        # 如果保存时包含了自定义层的某些键值，这里做一个简单的容错
        model = BertForSequenceClassification.from_pretrained(
            BEST_MODEL_PATH,
            output_attentions=True,
            ignore_mismatched_sizes=True
        )

    model.to(DEVICE)
    model.eval()

    # 加载数据
    if not os.path.exists(DATA_CACHE):
        print(f"Error: Data cache not found at {DATA_CACHE}")
        return

    dataset = load_from_disk(DATA_CACHE)
    val_dataset = dataset["validation"]
    print(f">> Dataset loaded. Validation size: {len(val_dataset)}")

    generated_files = []

    print(f">> Generating HTML for {NUM_SAMPLES} samples...")

    for i in range(NUM_SAMPLES):
        example = val_dataset[i]

        input_ids_tensor = torch.tensor(example["input_ids"]).unsqueeze(0).to(DEVICE)
        attention_mask_tensor = torch.tensor(example["attention_mask"]).unsqueeze(0).to(DEVICE)
        gold_mask_tensor = torch.tensor(example["gold_mask"]).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            # 提取最后一层，求平均
            last_layer_attn = outputs.attentions[-1]
            # Shape: (seq_len, )
            attn_score = last_layer_attn.mean(dim=1).squeeze(0).mean(dim=0)

        tokens_raw = tokenizer.convert_ids_to_tokens(example["input_ids"])
        attn_score_np = attn_score.cpu().numpy()
        gold_mask_np = gold_mask_tensor.cpu().numpy()

        # --- 关键步骤：过滤 Padding ---
        valid_indices = [idx for idx, t in enumerate(tokens_raw) if t != '[PAD]']

        filtered_tokens = [tokens_raw[idx] for idx in valid_indices]
        filtered_attn = attn_score_np[valid_indices]
        filtered_gold = gold_mask_np[valid_indices]

        # 生成 HTML 文件
        if len(filtered_tokens) > 0:
            fname = save_html_sample(i, filtered_tokens, filtered_attn, filtered_gold, VIS_DIR)
            generated_files.append(fname)
            print(f"  -> Generated: {fname}")

    # 生成总索引页
    if generated_files:
        create_index_html(generated_files, VIS_DIR)
        print(f"\n>> All Done! Open the following file in your browser to view results:")
        print(f"   {os.path.abspath(os.path.join(VIS_DIR, 'index.html'))}")
    else:
        print("No samples were generated.")


if __name__ == "__main__":
    main()