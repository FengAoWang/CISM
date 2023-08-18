import torch
import numpy as np
import torch.nn as nn


def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return indicator_matrix


def neg_par_log_likelihood(pred, ytime, yevent):
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

    return cost


def c_index(pred, ytime, yevent):
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i] = 1
            elif pred[i] == pred[j]:
                pred_matrix[j, i] = 0.5

    concord_matrix = pred_matrix.mul(ytime_matrix)
    concord = torch.sum(concord_matrix)
    epsilon = torch.sum(ytime_matrix)
    concordance_index = torch.div(concord, epsilon)
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
    return concordance_index


def CoxLoss(survtime, censor, hazard_pred):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


class KDFeatureLossTwo(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', alpha = 0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        # self.cross_entropy_clean = nn.CrossEntropyLoss(reduction = reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        # self.factor = factor
        self.alpha = alpha
        # self.beta = beta

    def forward(self, map_teacher1, map_student1, pred_noise, pred_clean, label):
        loss_ce_noise = self.cross_entropy(pred_noise, label)
        loss_ce_clean = self.cross_entropy(pred_clean, label)
        loss_map = self.l2_loss(map_teacher1, map_student1)
        loss = loss_ce_noise + loss_ce_clean + self.alpha*loss_map
        return loss


class KDLossAlignTwo(nn.Module):
    """ multi-label cross entropy loss """
    def __init__(self, reduction = 'mean', alpha = 0.1, beta=0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction = reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)
        self.alpha = alpha
        self.beta = beta

    def forward(self, mean_teacher, mean_student, map_teacher1, map_student1, map_teacher2, map_student2, pred_noise, pred_clean, label):
        loss_ce_noise = self.cross_entropy(pred_noise, label)
        loss_ce_clean = self.cross_entropy(pred_clean, label)
        loss_map_1 = self.l2_loss(map_teacher1, map_student1)
        loss_map_2 = self.l2_loss(map_teacher2, map_student2)
        loss_audio_mean = self.l2_loss(mean_teacher, mean_student)
        loss = loss_ce_noise  + self.alpha*loss_map_1 + self.beta*loss_map_2
        return loss
