import torch
import torch.nn as nn

class BPR(nn.Module):
    """
    Calculate predicted score
    """
    def __init__(self,user_num,item_num,factor_num):
        super(BPR,self).__init__()

        self.user_embedding = nn.Embedding(user_num,factor_num)
        self.item_embedding = nn.Embedding(item_num,factor_num)

        #normal distribution -> xavier_normal
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
    def forward(self,user,i,j):
        user = self.user_embedding(user)
        item_i = self.item_embedding(i)
        item_j = self.item_embedding(j)

        score_i = (user*item_i).sum(dim=-1)
        score_j = (user*item_j).sum(dim=-1)

        score = score_i - score_j
        return score
