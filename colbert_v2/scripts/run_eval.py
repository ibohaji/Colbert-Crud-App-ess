# colbert_v2/train/evaluator.py
from sklearn.metrics import ndcg_score


class Evaluator:
    @staticmethod
    def ndcg_score(true_labels, pred_scores, k=10):
        return ndcg_score(true_labels, pred_scores, k=k)
