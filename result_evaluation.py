import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from datasets import load_from_disk
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

CHECKPOINTS_DIR = "./checkpoints"
DATA_CACHE = "./data_cache/esnli_tokenized"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TOP_K_RATIO = 0.2

MODELS_CONFIG = [
    {"name": "Guided Model", "id": "mode_guided_lambda1.0_20260120_184228"},
    {"name": "Vanilla Model", "id": "mode_vanilla_lambda1.0_20260120_215539"},
    {"name": "Random Model", "id": "mode_random_lambda1.0_20260121_003525"},
]


class GuidedBertSelfAttention(BertSelfAttention):
    def __init__(self, config, lambda_guidance=1.0):
        super().__init__(config)
        self.lambda_guidance = lambda_guidance
        self.gold_mask = None

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = True
        gold_mask = kwargs.pop("gold_mask", self.gold_mask)
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


def get_faithfulness(model, input_ids, attention_mask, labels, last_attn, top_k_ratio=0.2):
    model.eval()
    with torch.no_grad():
        orig_logits = model(input_ids, attention_mask=attention_mask).logits
        orig_probs = F.softmax(orig_logits, dim=-1)
        target_probs = orig_probs.gather(1, labels.unsqueeze(-1)).squeeze()

        comp_scores = []
        suff_scores = []

        for i in range(input_ids.size(0)):
            seq_len = attention_mask[i].sum().item()
            k = max(1, int(seq_len * top_k_ratio))

            single_attn = last_attn[i][:seq_len]
            topk_indices = single_attn.topk(k).indices

            comp_input = input_ids[i].clone()
            comp_input[topk_indices] = 103
            c_logits = model(comp_input.unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0)).logits
            c_prob = F.softmax(c_logits, dim=-1)[0, labels[i]].item()
            comp_scores.append(target_probs[i].item() - c_prob)

            suff_input = torch.full_like(input_ids[i], 103)
            suff_input[topk_indices] = input_ids[i][topk_indices]
            suff_input[0] = input_ids[i][0]
            s_logits = model(suff_input.unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0)).logits
            s_prob = F.softmax(s_logits, dim=-1)[0, labels[i]].item()
            suff_scores.append(target_probs[i].item() - s_prob)

    return np.mean(comp_scores), np.mean(suff_scores)


def main():
    dataset = load_from_disk(DATA_CACHE)["validation"]
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    final_results = []

    for cfg in MODELS_CONFIG:
        path = os.path.join(CHECKPOINTS_DIR, cfg['id'], "best_model")
        print(f"\n>> Evaluating model: {cfg['name']}")

        model = BertForSequenceClassification.from_pretrained(path, num_labels=3, attn_implementation="eager")

        if "Guided" in cfg['name']:
            for layer in model.bert.encoder.layer[-4:]:
                new_attn = GuidedBertSelfAttention(model.config, 1.0)
                new_attn.load_state_dict(layer.attention.self.state_dict())
                layer.attention.self = new_attn

        model.to(DEVICE)
        model.eval()

        all_acc = []
        all_auprc = []
        all_comp = []
        all_suff = []

        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            gold_mask = batch["gold_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            if "Guided" in cfg['name']:
                for layer in model.bert.encoder.layer[-4:]:
                    layer.attention.self.gold_mask = gold_mask

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
                logits = outputs.logits
                last_attn = outputs.attentions[-1].mean(dim=1).mean(dim=1)

                preds = logits.argmax(dim=-1)
                all_acc.extend((preds == labels).cpu().numpy())

                batch_comp, batch_suff = get_faithfulness(model, input_ids, attention_mask, labels, last_attn,
                                                          TOP_K_RATIO)
                all_comp.append(batch_comp)
                all_suff.append(batch_suff)

                for b in range(input_ids.size(0)):
                    valid = (attention_mask[b] == 1).cpu().numpy()
                    y_true = batch["gold_mask"][b].numpy()[valid]
                    y_score = last_attn[b].cpu().numpy()[valid]
                    if y_true.sum() > 0:
                        all_auprc.append(average_precision_score(y_true, y_score))

        res = {
            "Model": cfg['name'],
            "Accuracy": round(np.mean(all_acc), 4),
            "AUPRC": round(np.mean(all_auprc), 4),
            "Comprehensiveness": round(np.mean(all_comp), 4),
            "Sufficiency": round(np.mean(all_suff), 4)
        }
        print(res)
        final_results.append(res)

    df = pd.DataFrame(final_results)
    df.to_csv("result_eval.csv", index=False)
    print("\n>> result_eval.csv")


if __name__ == "__main__":
    main()
