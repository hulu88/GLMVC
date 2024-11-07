import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
eps=1e-8

"""
# This code implements a custom Contrastive Loss class for multi-class contrastive learning tasks.
# Key features include:
# 1. **Masking Correlated Samples** : In order to avoid the effect of direct similarity between samples,
#    a mask matrix is defined to mask the similarity calculation of correlated samples.
# 2. **Forward Label (forward_label)** : Computes the contrastive loss, taking into account the entropy between classes
#   and the similarity between positive and negative samples. The loss function is optimized by calculating
#   the cosine similarity between positive and negative samples combined with the cross-entropy loss.
# 3. **Forward Label2 (forward_label2)** : Implements an improved version of contrastive loss,
#   which filters the similarity of positive samples and retains higher quality positive samples to further improve
#   the stability and performance of the model.
# 4. ** Cross-entropy Loss & Entropy Regularization ** : This method computes the cross-entropy of
#   the label probabilities and regularizes them in combination with information entropy to promote
#   better discrimination between classes.
"""


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device, margin):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.margin = margin
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask


    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i=torch.clamp(p_i,min=eps)
        ne_i =  (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        p_j=torch.clamp(p_j,min=eps)
        ne_j =  (p_j * torch.log(p_j)).sum()
        # entropy = ne_i + ne_j


        p_joint = (q_i * q_j).sum(0).view(-1)
        p_joint /= p_joint.sum()
        p_joint = torch.clamp(p_joint, min=eps)
        joint_entropy = (p_joint * torch.log(p_joint)).sum()
        entropy = ne_i + ne_j - joint_entropy

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_f
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)



        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        entropy/=N
        return loss+entropy

    def forward_label2(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        p_i = torch.clamp(p_i, min=eps)
        ne_i = (p_i * torch.log(p_i)).sum()

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # Screening of samples
        mean = positive_clusters.mean()
        std = positive_clusters.std()
        threshold = mean + std
        high_quality_clusters = positive_clusters[positive_clusters < threshold]
        if high_quality_clusters.dim() == 1:
            high_quality_clusters = high_quality_clusters.unsqueeze(1)
        negative_clusters = torch.zeros(len(high_quality_clusters), 1).to(high_quality_clusters.device)
        labels = torch.zeros(len(high_quality_clusters)).to(high_quality_clusters.device).long()
        logits = torch.cat((high_quality_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= len(high_quality_clusters)
        ne_i /= (N / 2)

        return loss+ne_i

