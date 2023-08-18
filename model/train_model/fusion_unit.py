import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from model.train_model.genomic_vae_model import Net_vae, set_bio_pathway
from model.train_model.ct_fe_vae_model import ct_fe_vae


def fusion_unit(fe1, fe2):
    fe1 = torch.cat((fe1, torch.FloatTensor(fe1.shape[0], 1).fill_(1).cuda()), dim=1).cuda()
    fe2 = torch.cat((fe2, torch.FloatTensor(fe2.shape[0], 1).fill_(1).cuda()), dim=1).cuda()
    fe12 = torch.bmm(fe1.unsqueeze(2), fe2.unsqueeze(1)).flatten(start_dim=1).cuda()
    return fe12


def mean_vector(fe1, fe2):
    fe = (fe1 + fe2) / 2
    return fe


genes = pd.read_csv('../data/ARGO_Data/gene_data/high_level_pathway.csv')
gene_dim = 2


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class FusionModel(nn.Module):
    def __init__(self, bio_model_path):
        super(FusionModel, self).__init__()
        self.risk_predictor = nn.Sequential(nn.Linear(729, 256),
                                            nn.BatchNorm1d(256),
                                            nn.ReLU(),
                                            nn.Linear(256, 64),
                                            nn.BatchNorm1d(64),
                                            nn.ReLU(),
                                            nn.Linear(64, 16),
                                            nn.BatchNorm1d(16),
                                            nn.ReLU(),
                                            nn.Linear(16, 1))
        self.mean_vector_predictor = nn.Sequential(nn.Linear(26, 1))

        self.genomic_to_ct = nn.Sequential(nn.Linear(26, 26), nn.ReLU())

        self.ct_to_genomic = nn.Sequential(nn.Linear(26, 26), nn.ReLU())

        self.ct_Discriminator = nn.Sequential(nn.Linear(26, 1), nn.Sigmoid())
        self.bio_model = Net_vae(len(genes), gene_dim)
        set_bio_pathway(self.bio_model, genes, gene_dim)
        self.bio_model.load_state_dict(torch.load(bio_model_path, map_location='cpu'))
        dfs_freeze(self.bio_model)

        self.genomic_Discriminator = nn.Sequential(nn.Linear(26, 1), nn.Sigmoid())

        self.ct_fc1 = nn.Linear(512, 256)
        self.ct_bn1 = nn.BatchNorm1d(256)

        self.ct_fc2 = nn.Linear(256, 128)
        self.ct_bn2 = nn.BatchNorm1d(128)
        self.ct_fc3 = nn.Linear(128, 26)

        self.fuse_fc = nn.Sequential(nn.Linear(26, 26), nn.ReLU())

    def forward(self, genomic_fe, ct_fe):
        genomic_fe = self.bio_model(genomic_fe)['z_sample_eq']
        ct_fe = F.relu(self.ct_bn1(self.ct_fc1(ct_fe)))
        ct_fe = F.relu(self.ct_bn2(self.ct_fc2(ct_fe)))
        ct_fe = self.ct_fc3(ct_fe)
        fusion_fe = fusion_unit(genomic_fe, ct_fe)
        # fusion_fe = self.fuse_fc(fusion_fe)
        risk = self.risk_predictor(fusion_fe)
        fusion_output = {'hazards': risk}
        return risk


class dual_vae_model(nn.Module):
    def __init__(self, bio_model_path, ct_model_path, two_stage=True):
        super(dual_vae_model, self).__init__()

        self.ct_fe_vae = ct_fe_vae(input_dim=512, hidden_dim=26)
        self.ct_fe_vae.load_state_dict(torch.load(ct_model_path, map_location='cpu'))

        self.bio_model = Net_vae(len(genes), gene_dim)
        set_bio_pathway(self.bio_model, genes, gene_dim)
        self.bio_model.load_state_dict(torch.load(bio_model_path, map_location='cpu'))
        self.two_stage = two_stage

        if two_stage:
            dfs_freeze(self.ct_fe_vae)
            dfs_freeze(self.bio_model)

        self.mean_vector_risk_predictor = nn.Sequential(nn.Linear(26, 26),
                                                        nn.ReLU(),
                                                        nn.Linear(26, 1))

        self.fusion_risk_predictor = nn.Sequential(nn.Linear(729, 256),
                                                   nn.BatchNorm1d(256),
                                                   nn.ReLU(),
                                                   nn.Linear(256, 64),
                                                   nn.BatchNorm1d(64),
                                                   nn.ReLU(),
                                                   nn.Linear(64, 16),
                                                   nn.BatchNorm1d(16),
                                                   nn.ReLU(),
                                                   nn.Linear(16, 1))

        self.genomic_to_ct = nn.Sequential(nn.Linear(26, 26), nn.ReLU())
        self.genomic_Discriminator = nn.Sequential(nn.Linear(26, 1), nn.Sigmoid())

        self.ct_to_genomic = nn.Sequential(nn.Linear(26, 26), nn.ReLU())
        self.ct_Discriminator = nn.Sequential(nn.Linear(26, 1), nn.Sigmoid())

    def forward(self, genomic_fe, ct_fe):
        genomic_output = self.bio_model(genomic_fe)
        genomic_fe = genomic_output['z_sample_eq']

        ct_output = self.ct_fe_vae(ct_fe)
        ct_fe = ct_output['z_sample_eq']
        ct_to_genomic_fe = self.ct_to_genomic(ct_fe)
        genomic_to_ct_fe = self.genomic_to_ct(genomic_fe)

        fusion_fe = mean_vector(genomic_fe, ct_fe)
        risk = self.mean_vector_risk_predictor(fusion_fe)
        fusion_output = {'hazards': risk, 'genomic_fe': genomic_fe, 'ct_fe': ct_fe, 'ct_to_genomic': ct_to_genomic_fe,
                         'genomic_to_ct': genomic_to_ct_fe}
        return fusion_output, genomic_output, ct_output
