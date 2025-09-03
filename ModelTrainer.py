from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ==== 1. Load the JSONL dataset ====
dataset = load_dataset("json", data_files=r"C:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG\training_dataset.jsonl", split="train")

# Convert each row into a valid input sample for trainer
train_examples = [
    InputExample(texts=[row["query"], row["positive"], row["negative"]])
    for row in dataset
]


model = SentenceTransformer("BAAI/bge-large-en-v1.5")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=3)

train_loss = losses.TripletLoss(model=model)

num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="bge_dense_finetuned_triplet",
    use_amp=True
)

print("Training complete. Model saved to `bge_dense_finetuned_triplet`.")