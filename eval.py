import numpy as np
import torch
from torchmetrics import AUROC

def hit(gt_item, pred_items):

    """whether gt_item is in pred_items or not"""
    
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for user, item_i, item_j in test_loader:
        user = user.to(device)
        item_i = item_i.to(device)
        item_j = item_j.to(device) # not useful when testing

        # prediction_i -> 전체 score를 고려할 수 있도록 수정
        score = model(user, item_i, item_j)
        _, indices = torch.topk(score, top_k)
        recommends = []
        for idx in indices:
            recommends.append(item_i[idx].cpu().numpy())
        
        gt_item = item_i[0].item()

        auroc = AUROC(task='binary')
        predict_auroc = auroc(recommends,gt_item)
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG), predict_auroc