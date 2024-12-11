# scripts/run_train.py
from colbert_v2 import Trainer, create_dataloader
from colbert_v2.models.indexer import Encoder

# Load data, create encoder and dataloader
texts, labels = load_your_data()  # Dummy function
dataloader = create_dataloader(texts, labels)

encoder = Encoder()
trainer = Trainer(encoder, dataloader)
trainer.train()

# scripts/run_eval.py
from colbert_v2 import Evaluator

# Dummy data
true_labels = ...
pred_scores = ...
print(Evaluator.ndcg_score(true_labels, pred_scores))
