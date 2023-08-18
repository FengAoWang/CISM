import torch
import torch.nn as nn
from model.train_model.genomic_model import Net, set_bio_pathway
from model.train_model.ct_fe_single_model import Rad_Net_omic
from model.train_model.fusion_model import fusion_unit
from model.train_model.newencoder import InterNetNew
import itertools
import numpy as np
import pandas as pd
from utils.pathway_module.pathways.reactome import Reactome, ReactomeNetwork
from torch.nn.utils import prune

genes = pd.read_csv('../data/ARGO_Data/gene_data/argo_exp_mut_cnv_match_genes.csv')
gene_dim = 2


class genomic_explain_model(nn.Module):
    def __init__(self, genes, gene_dim, n_genes, genomic_model_dict):
        super().__init__()
        self.genomic_net = Net(n_genes, gene_dim)
        set_bio_pathway(self.genomic_net, genes, gene_dim)
        self.genomic_net.load_state_dict(torch.load(genomic_model_dict, map_location='cpu'))

    def forward(self, x):
        output = self.genomic_net(x)
        return output['hazards']


class fusion_explain_unit(nn.Module):
    def __init__(self, fusion_model_path, genomic_model_path, ct_model_path):
        super(fusion_explain_unit, self).__init__()
        self.fusion_unit = fusion_unit(genomic_model_path, ct_model_path)

        self.fusion_unit.load_state_dict(torch.load(fusion_model_path, map_location='cpu'))
        # self.ct_model = Rad_Net_omic()
        # self.ct_model.load_state_dict(torch.load(ct_model_path, map_location='cpu'))

        # self.classifier = nn.Sequential(nn.Linear(26, 1))

    def forward(self, genomic_data, ct_image):
        genomic_features, f, hazards = self.fusion_unit(genomic_data, ct_image, 'two')
        # ct_fe, _ = self.ct_model(ct_image)
        # features = (genomic_fe + ct_fe) / 2
        # hazards = self.classifier(features)
        return hazards

