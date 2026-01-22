import os
import torch
import numpy as np
import html
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk

RUN_ID = "mode_random_lambda1.0_20260121_003525"
CHECKPOINTS_DIR = "../checkpoints"
RUN_DIR = os.path.join(CHECKPOINTS_DIR, RUN_ID)
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model")
DATA_CACHE = "./data_cache/esnli_tokenized"

VIS_DIR = os.path.join("./visualization_html", RUN_ID)
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 10


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


def get_color_style(score, is_gold=False):
    score = max(0.0, min(1.0, score))

    if score < 0.05:
        return "background-color: transparent; color: black;"

    if is_gold:
        r = 255
        g = int(255 * (1 - score))
        b = int(255 * (1 - score))
    else:
        r = int(255 * (1 - score))
        g = int(255 * (1 - score * 0.5))
        b = 255

    text_color = "white" if score > 0.7 else "black"

    return f"background-color: rgb({r}, {g}, {b}); color: {text_color};"


def save_html_sample(sample_id, tokens, attn_scores, gold_mask, output_dir):
    min_s = attn_scores.min()
    max_s = attn_scores.max()
    if max_s - min_s > 1e-9:
        norm_attn = (attn_scores - min_s) / (max_s - min_s)
    else:
        norm_attn = attn_scores

    filename = f"sample_{sample_id}.html"
    filepath = os.path.join(output_dir, filename)

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

    html_content += """
        <div class="card">
            <h3>🔹 Model Attention Prediction (Blue)</h3>
            <div class="text-box">
    """
    for token, raw_score, norm_score in zip(tokens, attn_scores, norm_attn):
        style = get_color_style(norm_score, is_gold=False)
        safe_token = html.escape(token)
        html_content += f'<span class="token" style="{style}" data-score="{raw_score:.4f}">{safe_token}</span>\n'
    html_content += "</div></div>"

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
        print("Warning: Standard loading failed, trying to ignore mismatched keys...")
        model = BertForSequenceClassification.from_pretrained(
            BEST_MODEL_PATH,
            output_attentions=True,
            ignore_mismatched_sizes=True
        )

    model.to(DEVICE)
    model.eval()

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
            last_layer_attn = outputs.attentions[-1]
            attn_score = last_layer_attn.mean(dim=1).squeeze(0).mean(dim=0)

        tokens_raw = tokenizer.convert_ids_to_tokens(example["input_ids"])
        attn_score_np = attn_score.cpu().numpy()
        gold_mask_np = gold_mask_tensor.cpu().numpy()

        valid_indices = [idx for idx, t in enumerate(tokens_raw) if t != '[PAD]']

        filtered_tokens = [tokens_raw[idx] for idx in valid_indices]
        filtered_attn = attn_score_np[valid_indices]
        filtered_gold = gold_mask_np[valid_indices]

        if len(filtered_tokens) > 0:
            fname = save_html_sample(i, filtered_tokens, filtered_attn, filtered_gold, VIS_DIR)
            generated_files.append(fname)
            print(f"  -> Generated: {fname}")

    if generated_files:
        create_index_html(generated_files, VIS_DIR)
        print(f"\n>> All Done! Open the following file in your browser to view results:")
        print(f"   {os.path.abspath(os.path.join(VIS_DIR, 'index.html'))}")
    else:
        print("No samples were generated.")


if __name__ == "__main__":
    main()