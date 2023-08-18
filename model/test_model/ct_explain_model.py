import torch
import torch.nn as nn
from model.train_model.genomic_model import Net, set_bio_pathway
from model.train_model.ct_fe_single_model import Rad_Net_omic
from model.train_model.newencoder import InterNetNew
from model.train_model.ct_fe_extractor import ct_fe_extractor
import pandas as pd

genes = pd.read_csv('../data/ARGO_Data/gene_data/argo_exp_mutation_genes_match.csv')
gene_dim = 2


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class ct_explain_model(nn.Module):
    def __init__(self, rad_net_dict):
        super(ct_explain_model, self).__init__()
        self.ct_fe_extractor = ct_fe_extractor()
        self.rad_net = Rad_Net_omic()
        self.rad_net.load_state_dict(torch.load(rad_net_dict, map_location='cpu'))

    def forward(self, x):
        fe = self.ct_fe_extractor(x)
        low_fe, hazard = self.rad_net(fe)
        return hazard


class fusion_unit(nn.Module):
    def __init__(self, genomic_model_path, ct_model_path, genomic_mean):
        super(fusion_unit, self).__init__()
        self.genomic_model = Net(len(genes), gene_dim)
        set_bio_pathway(self.genomic_model, genes, gene_dim)
        self.genomic_model.load_state_dict(torch.load(genomic_model_path, map_location='cpu'))
        self.ct_model = Rad_Net_omic()
        self.ct_model.load_state_dict(torch.load(ct_model_path, map_location='cpu'))

        self.encoder = InterNetNew(512)

        self.fc1 = nn.Sequential(nn.Linear(52, 26),
                                 nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(26, 1))
        self.softplus = nn.Softplus()
        self.genomic_mean = genomic_mean

    def forward(self, ct_image):
        ct_fe, _ = self.ct_model(ct_image)
        genomic_mean = self.genomic_mean.expand(ct_fe.shape[0], -1)

        weight = self.softplus(self.encoder(ct_image))
        genomic_features = genomic_mean.mul(weight)
        features = (ct_fe + genomic_features) / 2
        f = features
        hazards = self.classifier(features)
        return hazards


class explain_incomplete_network(nn.Module):
    def __init__(self, model_dict_path, genomic_model_path, ct_model_path, genomic_mean):
        super(explain_incomplete_network, self).__init__()
        self.ct_extractor = ct_fe_extractor()
        self.fusion_unit = fusion_unit(genomic_model_path, ct_model_path, genomic_mean)
        self.fusion_unit.load_state_dict(torch.load(model_dict_path, map_location='cpu'))

    def forward(self, ct_image):
        ct_fe = self.ct_extractor(ct_image)
        hazards = self.fusion_unit(ct_fe)
        return hazards
