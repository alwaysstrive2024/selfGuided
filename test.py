from datasets import load_dataset

# Load the e-SNLI dataset
dataset = load_dataset("esnli", trust_remote_code=True)

train_dataset = dataset['train']
eval_dataset = dataset['validation']