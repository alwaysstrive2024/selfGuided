import os
import torch
import torch.nn.functional as F
import numpy as np
import html
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk

MODELS_CONFIG = [
    {"name": "Guided Model", "id": "mode_guided_lambda1.0_20260120_184228", "color": "blue"},
    {"name": "Vanilla Model", "id": "mode_vanilla_lambda1.0_20260120_215539", "color": "green"},
    {"name": "Random Model", "id": "mode_random_lambda1.0_20260121_003525", "color": "purple"},
]

CHECKPOINTS_DIR = "./checkpoints"
DATA_CACHE = "./data_cache/esnli_tokenized"
VIS_DIR = "./visualization_full_detailed_report"
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
SAMPLES_PER_PAGE = 100
LABEL_MAP = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


class GuidedBertSelfAttention(BertSelfAttention):
    def __init__(self, config, lambda_guidance=1.0):
        super().__init__(config)
        self.lambda_guidance = lambda_guidance

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = True
        gold_mask = kwargs.pop("gold_mask", None)
        outputs = super().forward(*args, **kwargs)
        if gold_mask is not None:
            attn_probs = outputs[1]
            gm = gold_mask.unsqueeze(1).unsqueeze(2)
            guided = attn_probs * (1 + self.lambda_guidance * gm)
            guided = guided / (guided.sum(dim=-1, keepdim=True) + 1e-9)
            outputs = (outputs[0], guided) + outputs[2:]
        return outputs


def collate_fn(batch):
    return {k: torch.tensor([b[k] for b in batch]) for k in ["input_ids", "attention_mask", "gold_mask", "label"]}


def get_color_style(score, color_type='blue'):
    score = max(0.0, min(1.0, score))
    if score < 0.05: return "background-color: transparent; color: black;"
    if color_type == 'red':
        r, g, b = 255, int(255 * (1 - score)), int(255 * (1 - score))
    elif color_type == 'blue':
        r, g, b = int(255 * (1 - score)), int(255 * (1 - score * 0.6)), 255
    elif color_type == 'green':
        r, g, b = int(255 * (1 - score)), 255, int(255 * (1 - score))
    else:
        r, g, b = int(255 * (1 - score * 0.5)), int(255 * (1 - score)), 255
    text_color = "white" if score > 0.5 else "black"
    return f"background-color: rgb({r}, {g}, {b}); color: {text_color};"


def main():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_from_disk(DATA_CACHE)["validation"]
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_results = []
    for i in range(len(dataset)):
        all_results.append({
            "id": i,
            "true_label": LABEL_MAP[dataset[i]['label']],
            "explanation": dataset[i].get('explanation_1', 'N/A'),
            "tokens": tokenizer.convert_ids_to_tokens(dataset[i]['input_ids']),
            "gold_mask": dataset[i]['gold_mask'],
            "model_outputs": []
        })

    for m_cfg in MODELS_CONFIG:
        path = os.path.join(CHECKPOINTS_DIR, m_cfg['id'], "best_model")
        print(f"\n>> Inferencing with {m_cfg['name']}...")
        model = BertForSequenceClassification.from_pretrained(path, output_attentions=True).to(DEVICE)
        model.eval()

        ptr = 0
        for batch in tqdm(val_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                out = model(**inputs)
                probs = F.softmax(out.logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)
                attns = out.attentions[-1].mean(dim=1).mean(dim=1)  # (batch, seq)

            batch_size = inputs['input_ids'].size(0)
            for b in range(batch_size):
                all_results[ptr + b]["model_outputs"].append({
                    "name": m_cfg['name'],
                    "pred": LABEL_MAP[preds[b].item()],
                    "conf": confs[b].item(),
                    "attn": attns[b].cpu().numpy(),
                    "color": m_cfg['color'],
                    "is_correct": preds[b].item() == batch['label'][b].item()
                })
            ptr += batch_size

        del model
        torch.cuda.empty_cache()

    total_samples = len(all_results)
    total_pages = math.ceil(total_samples / SAMPLES_PER_PAGE)
    print(f"\n>> Generating {total_pages} HTML pages...")

    for p in range(1, total_pages + 1):
        start = (p - 1) * SAMPLES_PER_PAGE
        end = min(start + SAMPLES_PER_PAGE, total_samples)

        page_body = ""
        for i in range(start, end):
            res = all_results[i]
            valid_idx = [idx for idx, t in enumerate(res['tokens']) if t != '[PAD]']
            tokens = [res['tokens'][idx] for idx in valid_idx]

            rows_html = ""
            for m_out in res['model_outputs']:
                f_attn = m_out['attn'][valid_idx]
                norm_attn = (f_attn - f_attn.min()) / (f_attn.max() - f_attn.min() + 1e-9)
                t_html = ""
                for t, raw_s, norm_s in zip(tokens, f_attn, norm_attn):
                    t_html += f'<span class="token" style="{get_color_style(norm_s, m_out["color"])}" title="Score: {raw_s:.4f}">{html.escape(t)}</span>'

                rows_html += f"""
                                <div class="row">
                                    <div class="meta"><strong>{m_out['name']}</strong><br>
                                        <span class="{'correct' if m_out['is_correct'] else 'wrong'}">{m_out['pred']} ({m_out['conf']:.1%})</span>
                                    </div>
                                    <div class="text">{t_html}</div>
                                </div>"""

            gold_attn = np.array(res['gold_mask'])[valid_idx]
            g_html = "".join([f'<span class="token" style="{get_color_style(s, "red")}">{html.escape(t)}</span>'
                              for t, s in zip(tokens, gold_attn)])

            page_body += f"""
            <div class="card">
                <div class="card-title">Sample #{i} | <small>True: {res['true_label']}</small></div>
                <div class="expl">Rationale: {html.escape(res['explanation'])}</div>
                {rows_html}
                <div class="row"><div class="meta"><strong>HUMAN GOLD</strong></div><div class="text">{g_html}</div></div>
            </div>"""

        nav = "".join([f'<a href="page_{k}.html" class="{"active" if k == p else ""}">{k}</a>' for k in
                       range(1, total_pages + 1)])

        full_html = f"""<html><head><meta charset="UTF-8"><style>
            body {{ font-family: sans-serif; background:#f0f2f5; padding:20px; }}
            .card {{ background:white; border-radius:12px; padding:20px; margin-bottom:30px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
            .card-title {{ font-size:1.2em; font-weight:bold; border-bottom:1px solid #eee; padding-bottom:10px; }}
            .row {{ display:flex; border-bottom:1px solid #f8f8f8; padding:8px 0; }}
            .meta {{ width:180px; font-size:0.85em; background:#fcfcfc; padding:5px; flex-shrink:0; }}
            .text {{ flex:1; padding-left:15px; line-height:2.2; }}
            .token {{ display:inline-block; padding:0 3px; border-radius:3px; font-size:14px; margin:1px; }}
            .expl {{ color:#777; font-style:italic; margin:10px 0; font-size:0.9em; }}
            .correct {{ color:green; font-weight:bold; }} .wrong {{ color:red; font-weight:bold; }}
            .nav {{ text-align:center; margin:20px 0; sticky; top:0; background:rgba(255,255,255,0.9); padding:10px; border-radius:8px; }}
            .nav a {{ margin:0 3px; text-decoration:none; color:#333; padding:2px 6px; border:1px solid #ddd; border-radius:3px; font-size:12px; }}
            .nav a.active {{ background:#007bff; color:white; border-color:#007bff; }}
        </style></head><body>
        <div class="nav"><strong>Pages:</strong> {nav}</div>
        <div style="max-width:1100px; margin:auto;">{page_body}</div>
        <div class="nav">{nav}</div>
        </body></html>"""

        with open(os.path.join(VIS_DIR, f"page_{p}.html"), "w", encoding="utf-8") as f:
            f.write(full_html)

    print(f"\nfinal report has been generated {VIS_DIR}")


if __name__ == "__main__":
    main()