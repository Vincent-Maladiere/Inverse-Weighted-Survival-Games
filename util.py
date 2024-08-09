import models
import os
import copy
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter as KMFitter
import numpy as np

# local
import catdist
import data_utils
import _concordance
import _nll
import _km



def str_to_bool(arg):
    """Convert an argument string into its boolean value.
    Args:
        arg: String representing a bool.
    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def isnan(x):
    return torch.any(torch.isnan(x))
 
def safe_log(x,eps):
    return (x+eps).log()

def clip(prob,clip_min):
    return prob.clamp(min=clip_min)

def round3(x):
    return round(x,3)

class Meter:
    def __init__(self):
        self.N = 0
        self.total = 0
    def update(self,val,N):
        self.total += val
        self.N += N
    def avg(self):
        return round(self.total / self.N,4)

def X_to_dist(X,model,args,k=None):
    pred_params = model(X)
    return catdist.CatDist(logits=pred_params, args=args, probs=None,k=k)

def X_to_FG_dists(X,Fmodel,Gmodel,args,k=None):
    Fdist = X_to_dist(X,Fmodel,args,k=k)
    Gdist = X_to_dist(X,Gmodel,args,k=k)
    return Fdist,Gdist


############################################
############ IPCW BS and BLL ###############
############################################


def IPCW_batch(fn,k,tgt,Fdist,Gdist,args,is_g=False,detach=True):
   
    if is_g:
        numer_dist=Gdist
        denom_dist=Fdist
    else:
        numer_dist=Fdist
        denom_dist=Gdist
    
    U,Delta=tgt
    kbatch = torch.ones_like(U) * k

    ncdf_k = numer_dist.leq(kbatch)
    observed = ~Delta if is_g else Delta

    if fn == 'bll_game':
        left_loss = -1.0 * safe_log(ncdf_k,args.logeps)
        right_loss = -1.0 * safe_log(1. - ncdf_k,args.logeps)
    elif fn == 'bs_game':
        left_loss = (1. - ncdf_k).pow(2)
        right_loss = ncdf_k.pow(2)
    else:
        assert False

    left_numer = left_loss * observed * (U <= kbatch)
    
    if is_g:
        left_denom = denom_dist.gt(U)
    else:
        left_denom = denom_dist.geq(U)
    left_denom = clip(left_denom, args.clip_min)

    right_numer = right_loss * (U > kbatch)
    right_denom = clip(denom_dist.gt(kbatch),args.clip_min)

    if detach:
        left_denom = left_denom.detach()
        right_denom = right_denom.detach()

    left = left_numer / left_denom
    right = right_numer / right_denom
    ipcw_loss = (left + right).mean(0)
    return ipcw_loss



def uncensored_BS_or_BLL_batch(fn,k,U,Fdist,args):
    kbatch = torch.ones_like(U) * k
    Fk = Fdist.cdf(kbatch)
    if fn=='bs_game':
        # BS(k) = E_T  [  1[T <= k] * (1-F(k))^2 + F(k)^2 1[T>k]  ]
        loss_k = torch.where(U <= kbatch, (1-Fk).pow(2), Fk.pow(2))
    else:
        # BS(k) = E_T  [  1[T <= k] * (1-F(k))^2 + F(k)^2 1[T>k]  ]
        loss_k = -1.0 * torch.where(U <= kbatch, safe_log(Fk,args.logeps), safe_log(1-Fk,args.logeps))

    assert loss_k.shape[0]==U.shape[0]
    loss_k = loss_k.mean(0)
    return loss_k

def game(fn, phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None, mode='normal'):
    return cond_bs_game(fn, phase, loader, Fmodel, Gmodel, args, Foptimizer=Foptimizer, Goptimizer=Goptimizer, mode=mode)


def cond_bs_game(fn, phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None, mode='normal'):
    Fsumm = 0.0
    Gsumm = 0.0

    for k in range(args.K-1):
        floss_meter_k = Meter()
        gloss_meter_k = Meter()
        for batch_idx, batch in enumerate(loader):     
            (U,_,Delta,X) = batch
            U=U.to(args.device)
            Delta=Delta.to(args.device)
            X=X.to(args.device)  
            bsz = U.shape[0]
            if phase=='train':
                Foptimizer.zero_grad()
                Goptimizer.zero_grad()
                Fdist,Gdist = X_to_FG_dists(X, Fmodel, Gmodel, args, k=None)
            else:
                Fdist,Gdist = X_to_FG_dists(X, Fmodel, Gmodel, args, k=None)
            if mode=='normal':
                floss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, args, is_g=False, detach=True)
                gloss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, args, is_g=True, detach=True)
            elif mode=='uncensored':
                Fdist = X_to_dist(X, Fmodel, args, k=None)
                floss_k = uncensored_BS_or_BLL_batch(fn, k, U, Fdist, args)
                gloss_k = torch.tensor([-1.0])
            elif mode=='kmG':
                assert phase=='test'
                Fdist = X_to_dist(X, Fmodel, args, k=None)
                G_cdfvals = _km.get_KM_cdfvals(loader, args)
                Gdist = _km.cdfvals_to_dist(G_cdfvals, bsz, args)
                floss_k = IPCW_batch(fn, k, (U,Delta), Fdist, Gdist, args=args, is_g=False, detach=True)
                gloss_k = torch.tensor([-1.0]).to(args.device)
            else:
                assert False

            if phase=='train':
                floss_k.backward()
                Foptimizer.step()
                gloss_k.backward()
                Goptimizer.step()
            floss_meter_k.update(val = floss_k.item() *  bsz, N = bsz)
            gloss_meter_k.update(val = gloss_k.item() *  bsz, N = bsz)
        Fsumm += floss_meter_k.avg()
        Gsumm += gloss_meter_k.avg()
    Fsumm = Fsumm / (args.K-1)
    Gsumm = Gsumm / (args.K-1)
    return Fsumm,Gsumm

def train_or_val(phase, loader, Fmodel, Gmodel, args, Foptimizer=None, Goptimizer=None):
   
    if phase=='train':
        assert Foptimizer is not None
        assert Goptimizer is not None

    if args.loss_fn == 'nll':
        floss,gloss = _nll.nll_FG(phase, loader, Fmodel, Gmodel, args, Foptimizer, Goptimizer)
    elif args.loss_fn in ['bs_game', 'bll_game']:
        floss,gloss = game(args.loss_fn, phase, loader, Fmodel, Gmodel, args, Foptimizer, Goptimizer)
    else:
        assert False

    return floss,gloss


# for game eval
def get_model_from_file(fname,args):
    ckpt = torch.load(fname,map_location=args.device)
    model = args.model_fn(args)
    model.load_state_dict(ckpt['model_state'])
    model.to(args.device)
    model.eval()
    return model

def eval_next_dist(cur_F_file, cur_G_file,args, get_F,loader,fn):
    
    # all models
    dirr = os.path.join(args.save_dir,args.ckpt_basename)
    F_filenames = [os.path.join(dirr,args.ckpt_basename+'_F_epoch{}.pth.tar'.format(epoch)) for epoch in range(0,args.epochs)]
    G_filenames = [os.path.join(dirr,args.ckpt_basename+'_G_epoch{}.pth.tar'.format(epoch)) for epoch in range(0,args.epochs)]

    # cur models
    cur_F = get_model_from_file(cur_F_file,args)
    cur_G = get_model_from_file(cur_G_file,args)

    best_metric = 1000000.0
    best_filename = None
    
    # given cur G, find an F
    if get_F:
        print("Searching for a new F")
        for candidate_model_filename in F_filenames:
            #print("--- Trying: {}".format(candidate_model_filename))
            candidate_model = get_model_from_file(candidate_model_filename,args)
            floss,_ = game(fn,'valid',loader,candidate_model,cur_G,args)
            if floss < best_metric:
                best_metric = floss
                best_filename = candidate_model_filename
        if best_filename==cur_F_file:
            changed=False
        else:
            changed=True


    # given cur F, find a G
    else:
        print("Searching for a new G")
        for candidate_model_filename in G_filenames:
            #print("--- Trying: {}".format(candidate_model_filename))
            candidate_model = get_model_from_file(candidate_model_filename,args)
            _,gloss = game(fn,'valid',loader,cur_F,candidate_model,args)
            if gloss < best_metric:
                best_metric = gloss
                best_filename = candidate_model_filename
        if best_filename==cur_G_file:
            changed=False
        else:
            changed=True

    return best_filename, changed






def f_metrics(loaders,Fmodel,Gmodel,args):

    if args.dataset in ['gamma','mnist']:
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders

    if args.dataset in ['gamma','mnist']:
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders
    elif args.dataset in args.realsets:
        trainloader,valloader,testloader = loaders
    else:              
        assert False 

    fbs_km,_ = game('bs_game','test',testloader,Fmodel,Gmodel,args,mode='kmG')
    fbs_ipcw,_= game('bs_game','test',testloader,Fmodel,Gmodel,args)
    fbll_km,_ = game('bll_game','test',testloader,Fmodel,Gmodel,args,mode='kmG')
    fbll_ipcw,_ = game('bll_game','test',testloader,Fmodel,Gmodel,args)
    fnll,_ = _nll.nll_FG('test',testloader,Fmodel,Gmodel,args)         
    fconc,_ = _concordance.test_concordance_FG(testloader,testloader,Fmodel,Gmodel,args)
    
    y_train, y_test, y_pred, time_grid, n_features = get_targets(trainloader, testloader, Fmodel, args)

    ### Start of hazardous code snippet ###
    from hazardous.metrics._yana import CensoredNegativeLogLikelihoodSimple

    y_pred = y_pred[None, :, :]
    y_surv = 1 - y_pred
    y_pred = np.concatenate([y_surv, y_pred], axis=0)

    censlog = CensoredNegativeLogLikelihoodSimple().loss(
        y_pred, y_test["duration"], y_test["event"], time_grid
    )

    from hazardous.metrics._brier_score import integrated_brier_score_incidence
    from hazardous.metrics._brier_score import brier_score_incidence

    event_id = 1

    ibs = integrated_brier_score_incidence(
        y_train,
        y_test,
        y_pred[event_id],
        times=time_grid,
        event_of_interest="any",
    )
    brier_scores = brier_score_incidence(
        y_train,
        y_test,
        y_pred[event_id],
        times=time_grid,
        event_of_interest="any",
    )
    event_specific_ibs = [{
        "event": event_id,
        "ibs": round(ibs, 4),
    }]
    event_specific_brier_scores = [{
        "event": event_id,
        "time": list(time_grid.round(2)),
        "brier_score": list(brier_scores.round(4)),
    }]

    horizons = [.25, .50, .75]
    c_indices = get_c_index(y_train, y_test, y_pred[event_id], time_grid, horizons, args)
    event_specific_c_index = [
        {
            "event": event_id,
            "time_quantile": horizons,
            "c_index": c_indices,
        }
    ]

    # MSE, MAE, D-Calibration
    from SurvivalEVAL import SurvivalEvaluator

    evaluator = SurvivalEvaluator(
        predicted_survival_curves=y_pred[0, :, :],
        time_coordinates=time_grid,
        test_event_indicators=y_test["event"].to_numpy(),
        test_event_times=y_test["duration"].to_numpy(),
        train_event_indicators=y_train["event"].to_numpy(),
        train_event_times=y_train["duration"].to_numpy(),
    )
    mse = evaluator.mse(method="Pseudo_obs")
    mae = evaluator.mae(method="Pseudo_obs")
    auc = evaluator.auc() 

    if args.dataset in ['gamma','mnist']:
        assert torch.all(torch.eq(Ftestloader.dataset.Delta,1))
        fbs_uncensored,_ = game('bs_game','test',Ftestloader,Fmodel,Gmodel,args,mode='uncensored')
        print("test fbs uncensored",fbs_uncensored)
        fbll_uncensored,_ = game('bll_game','test',Ftestloader,Fmodel,Gmodel,args,mode='uncensored')
    else:
        fbs_uncensored = -1.0
        fbll_uncensored = -1.0

    metrics = {}     
    metrics['fnll']=round3(fnll)             
    metrics['fconc']=round3(fconc)             
    metrics['fbs_km']=round3(fbs_km)          
    metrics['fbll_km']=round3(fbll_km)      
    metrics['fbs_ipcw']=round3(fbs_ipcw)       
    metrics['fbll_ipcw']=round3(fbll_ipcw)  
    metrics['fbs_uncensored']=round3(fbs_uncensored)
    metrics['fbll_uncensored']=round3(fbll_uncensored)

    import json
    from pathlib import Path

    model_name = f"han-{args.loss_fn}"
    scores = {
        "is_competing_risk": False,
        "n_events": 1,
        "model_name": model_name,
        "dataset_name": args.dataset,
        "n_rows": y_train.shape[0],
        "n_cols": n_features,
        "censoring_rate": y_train["event"].mean().round(4),
        "random_state": args.random_state,
        "time_grid": np.asarray(time_grid, dtype="float32").tolist(),
        "y_pred": np.asarray(y_pred, dtype="float32").tolist(),
        "predict_time": None,
        "event_specific_ibs": event_specific_ibs,
        "event_specific_brier_scores": event_specific_brier_scores,
        "event_specific_c_index": event_specific_c_index,
        "censlog": censlog,
        "mse": mse,
        "mae": mae,
        "auc": auc,
        "fit_time": args.fit_time,
    }

    path_dir = Path("../benchmark/scores") / "raw" / model_name
    path_dir.mkdir(parents=True, exist_ok=True)
    path_file = path_dir /  f"{args.dataset}.json"

    if path_file.exists():
        all_scores = json.load(open(path_file))
    else:
        all_scores = []
    
    all_scores.append(scores)
    json.dump(all_scores, open(path_file, "w"))
    print(f"Wrote {path_file}")

    ### End of hazardous code snippet

    return metrics


def eval_nll(loaders,Fmodel,Gmodel,args):
    return f_metrics(loaders,Fmodel,Gmodel,args)

def eval_game(loaders,args, tic):

    if args.dataset in ['gamma','mnist']:
        trainloader,valloader,testloader,Ftestloader,Gtestloader = loaders
    elif args.dataset in args.realsets:
        trainloader,valloader,testloader = loaders
    else:
        assert False

    dirr = os.path.join(args.save_dir,args.ckpt_basename)
    cur_F_file = os.path.join(dirr,args.ckpt_basename+'_F_epoch0.pth.tar')
    cur_G_file = os.path.join(dirr,args.ckpt_basename+'_G_epoch0.pth.tar')
    while True:
        print("Trying")
        cur_F_file, F_changed = eval_next_dist(cur_F_file,cur_G_file,args,get_F=True,loader=valloader,fn='bs_game')
        cur_G_file, G_changed = eval_next_dist(cur_F_file,cur_G_file,args,get_F=False,loader=valloader,fn='bs_game')
        if not F_changed and not G_changed:
            break
   
    best_F = get_model_from_file(cur_F_file,args)
    best_G = get_model_from_file(cur_G_file,args)
    import time
    toc = time.time()
    args.fit_time = toc - tic
    print(f"time to train: {toc - tic:.2f} seconds")
    return f_metrics(loaders,best_F,best_G,args), cur_F_file, cur_G_file


def get_bin_boundaries(times, K):
    percents = np.arange(K) * 100. / K
    return np.percentile(times, percents)
    

def get_targets(trainloader, testloader, model, args):
    all_y_proba = []
    all_discrete_duration = []
    all_duration = []
    all_event = []
    for (U, cU, Delta, X) in testloader:
        y_proba = model(X)
        all_y_proba.append(y_proba)
        all_discrete_duration.append(U)
        all_duration.append(
            cU.detach().numpy()
        )
        all_event.append(
            Delta.detach().numpy()
        )

    all_discrete_duration = torch.tensor(np.hstack(all_discrete_duration))
    all_duration = np.hstack(all_duration)
    all_event = np.hstack(all_event)

    all_y_proba = torch.tensor(np.concatenate(all_y_proba, axis=0))
    risk = catdist.CatDist(all_y_proba, args).probs.cumsum(-1)
    risk = risk.detach().numpy()

    # import torch.nn.functional as F
    # hazard = F.softplus(all_y_proba)
    # hazard = hazard.detach().numpy()
    # surv = np.exp(-hazard.cumsum(axis=1))
    # risk = 1 - surv

    import pandas as pd
    y_test = pd.DataFrame(
        dict(
            event=all_event,
            duration=all_duration,
        )
    )

    all_duration = []
    all_event = [] 
    for (_, cU, Delta, _) in trainloader:
        all_duration.append(cU)
        all_event.append(Delta)
    all_duration = np.hstack(all_duration)
    all_event = np.hstack(all_event)

    y_train = pd.DataFrame(
        dict(
            event=all_event,
            duration=all_duration,
        )
    )

    time_grid = get_bin_boundaries(all_duration[all_event], args.K)

    return y_train, y_test, risk, time_grid, X.shape[1]


def make_recarray(y):
    event = y["event"].to_numpy()
    duration = y["duration"].to_numpy()
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


def get_c_index(y_train, y_test, y_pred, time_grid, horizons, args):

    from sksurv.metrics import concordance_index_ipcw

    taus = np.quantile(time_grid, horizons)

    if args.N_test_c_index is not None:
        from sklearn.model_selection import train_test_split

        y_test = y_test.reset_index(drop=True)
        y_test, _ = train_test_split(
            y_test,
            stratify=y_test["event"],
            train_size=args.N_test_c_index,
            shuffle=True,
            random_state=args.random_state,
        )
        y_pred = y_pred[y_test.index, :]

    et_train = make_recarray(y_train)
    et_test = make_recarray(y_test)

    c_indexes = []
    from tqdm import tqdm
    for tau in tqdm(taus, desc="computing c-index"):
        idx_tau = np.searchsorted(time_grid, tau)
        y_pred_at_t = y_pred[:, idx_tau]
        
        print(y_pred_at_t.shape, y_train.shape, y_test.shape)
        ct_index, _, _, _, _ = concordance_index_ipcw(
            et_train,
            et_test,
            y_pred_at_t,
            tau=tau,
        )
        c_indexes.append(round(ct_index, 3))
    
    return c_indexes
