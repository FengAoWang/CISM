import torch
import numpy as np
import os
from sksurv.metrics import concordance_index_censored
from ..loss_function.surv_loss_func import NLLSurvLoss


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None,
                      loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic)  # return hazards, S, Y_hat, A_raw, results_dict

        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            h_path, h_omic, h_mm = h
            loss = 0.5 * loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.25 * loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25 * loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            h = h_mm

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival(model, loader, n_classes, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data) in enumerate(loader):
        dss_status, os_months, inputs, disc_label = data
        inputs = inputs[mode].to(device)
        os_months = os_months.to(device)
        dss_status = dss_status.to(device)
        disc_label = disc_label.to(device)

        with torch.no_grad():
            h = model(inputs)['hazards']

        if isinstance(h, tuple):
            h = h[2]

        if h.shape[1] > 1:
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        event_time = np.asscalar(os_months)
        censor = np.asscalar(dss_status)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return c_index


def get_risk(h):
    if isinstance(h, tuple):
        h = h[2]
    if h.shape[1] > 1:
        hazards = torch.sigmoid(h)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    else:
        risk = h.detach().cpu().numpy().squeeze()
    return risk


def summary_survival_fusion(model, loader, n_classes, mode):
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    single_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data) in enumerate(loader):
        dss_status, os_months, inputs, disc_label = data

        os_months = os_months.cuda()
        dss_status = dss_status.cuda()
        disc_label = disc_label.cuda()

        with torch.no_grad():
            output = model(inputs)
        fusion_modal_risk = get_risk(output['poe_output']['hazards'])
        single_modal_risk = get_risk(output[mode]['hazards'])
        event_time = np.asscalar(os_months)
        censor = np.asscalar(dss_status)
        all_risk_scores[batch_idx] = fusion_modal_risk
        single_risk_scores[batch_idx] = single_modal_risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time

    fusion_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    single_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, single_risk_scores, tied_tol=1e-08)[0]
    return fusion_c_index, single_c_index


def summary_survival_multimodal(model, loader, n_classes):
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    ct_risk_scores = np.zeros((len(loader)))
    genomic_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data) in enumerate(loader):
        dss_status, os_months, inputs, disc_label = data

        os_months = os_months.cuda()
        dss_status = dss_status.cuda()
        disc_label = disc_label.cuda()

        with torch.no_grad():
            output = model(inputs)
        fusion_modal_risk = get_risk(output['product_output']['hazards'])
        ct_modal_risk = get_risk(output['ct_output']['hazards'])
        genomic_modal_risk = get_risk(output['genomic_output']['hazards'])

        event_time = np.asscalar(os_months)
        censor = np.asscalar(dss_status)
        all_risk_scores[batch_idx] = fusion_modal_risk
        ct_risk_scores[batch_idx] = ct_modal_risk
        genomic_risk_scores[batch_idx] = genomic_modal_risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time

    fusion_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    ct_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, ct_risk_scores, tied_tol=1e-08)[0]
    genomic_c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, genomic_risk_scores, tied_tol=1e-08)[0]

    return fusion_c_index, ct_c_index, genomic_c_index


