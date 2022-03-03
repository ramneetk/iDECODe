# python ood_audio/check_ood_non_conf.py --training_id class_set_$0/1/2/3$ --class_set_name $0/1/2/3$ --n $1 for ICAD, 5 for ours with |V(x)| = 5 and 20 for SBP, base score and ours with |V(x)| = 20$ --trials 5

import json
import pickle
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats

import cli

import glob
import os.path

import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import inference
import pytorch.models as models
import pytorch.utils as utils
from pytorch.utils import ImageLoader
from pytorch.utils import Logger

from datasets import Datasets

from pytorch.specaug import SpecAugment

import torch.nn.functional as F
import argparse

import pdb

def _mask(arg):
    """Convert the ``--mask`` argument.

    The string must be in the format ``key1=value1,key2=value2,...``,
    and is converted into a Python dict.
    """
    if not arg:
        return dict()
    return {k: int(v) for k, v in [spec.split('=') for spec in arg.split(',')]}

parser = argparse.ArgumentParser()
parser.add_argument('--extraction_path', default='_workspace/features', help=' path to the directory containing extracted feature vectors')
parser.add_argument('--block_size', default=128, type=int)
parser.add_argument('--mask', default='manually_verified=1', type=_mask, help='for manually verified training samples')

parser.add_argument('--model_path', default='_workspace/models', help=' path to the saved model')
parser.add_argument('--dataset_path', default='_dataset', help=' path to the dataset')
parser.add_argument('--training_id')
parser.add_argument('--epochs', default=29, type=int, help='epoch no of the trained model')

# in-dist class name
parser.add_argument('--class_set_name', default=0, type=int)
parser.add_argument('--n', type=int, default=20, help='no. of transformations')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for taking average for the final results')

args = parser.parse_args()

datasets = Datasets(args.dataset_path)

def get_orig_data():
    """store predictions of the trained model on the in-dist test, cal and ood set
    """
    import pytorch.training as training
    import relabel
    import utils

    # Load training data and metadata

    x_train, df_train = _load_features(
        datasets.get('training'), args.extraction_path, args.block_size)

    # Use subset of training set (validation set) as calibration set
    mask = df_train.validation == 1
    x_cal = x_train[mask]
    df_cal = df_train[mask]
    x_train = x_train[~mask]
    df_train = df_train[~mask]
    
    in_dist_class_set_labels = utils.LABELS[5*args.class_set_name:(5*args.class_set_name)+5]
    print("in_dist_class_set_labels: ", in_dist_class_set_labels)
    x_cal = x_cal[df_cal.label.isin(in_dist_class_set_labels)]
    df_cal = df_cal[df_cal.label.isin(in_dist_class_set_labels)]

    if args.mask:
        x_cal, df_cal = _mask_data(x_cal, df_cal, args.mask)

    # get test samples = test samples on the in-dist class
    x_test, df_test = _load_features(datasets.get('test'), args.extraction_path, args.block_size)
    x_ood= x_test[~df_test.label.isin(in_dist_class_set_labels)]
    df_ood = df_test[~df_test.label.isin(in_dist_class_set_labels)]
    x_test = x_test[df_test.label.isin(in_dist_class_set_labels)]
    df_test = df_test[df_test.label.isin(in_dist_class_set_labels)]
    
    # saves predictions on orig (without transformation) test, cal and oods set
    # predict_on_orig_data(x_test, df_test, args, 'pred_in_dist_test')
    # predict_on_orig_data(x_cal, df_cal, args, 'pred_cal')
    # predict_on_orig_data(x_ood, df_ood, args, 'pred_ood')

    return x_test, df_test, x_cal, df_cal, x_ood, df_ood # orig (without transformation)

def _load_features(dataset, data_path, block_size=128):
    """Load the features and the associated metadata for a dataset.

    The metadata is read from a CSV file and returned as a DataFrame.
    Each DataFrame entry corresponds to an instance in the dataset.

    Args:
        dataset (Dataset): Information about the dataset.
        data_path (str): Path to directory containing feature vectors.

    Returns:
        tuple: Tuple containing the array of feature vectors and the
        metadata of the dataset.
    """
    import features
    import utils

    # Load feature vectors from disk
    features_path = os.path.join(data_path, dataset.name + '.h5')
    x, n_blocks = utils.timeit(lambda: features.load_features(features_path,
                                                              block_size,
                                                              block_size // 4),
                               f'Loaded features of {dataset.name} dataset')
    # Reshape feature vectors: NxTxF -> NxTxFx1
    x = np.expand_dims(x, axis=-1)

    # Load metadata and duplicate entries based on number of blocks
    df = pd.read_csv(dataset.metadata_path, index_col=0)
    df = df.loc[np.repeat(df.index, n_blocks)]

    return x, df

def _mask_data(x, df, specs):
    """Mask data using the given specifications.

    Args:
        x (array_like): Array of data to mask.
        df (pd.DataFrame): Metadata used to apply the specifications.
        specs (dict): Specifications used to mask the data.
    """
    mask = np.ones(len(df), dtype=bool)
    for k, v in specs.items():
        if k[-1] == '!':
            mask &= df[k[:-1]] != v
        else:
            mask &= df[k] == v
    return x[mask], df[mask]

def predict(x, df, epoch, model_path, orig, batch_size=128):
    """Compute predictions using a saved model.
    The model that was saved after the specified epoch is used to
    compute predictions. After block-level predictions are computed,
    they are merged to give clip-level predictions.
    Args:
        x (np.ndarray): Array of input data.
        df (pd.DataFrame): Associated metadata.
        epoch (int): Epoch number of the model to load.
        model_path (str): Path to directory containing saved models.
        batch_size (int): Number of instances to predict per batch.
        callback: Optional callback used for inference.
    Returns:
        np.ndarray: The clip-level predictions.
    """

    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model from disk
    model_path = os.path.join(model_path, f'model.{epoch:02d}.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model = models.create_model(*checkpoint['creation_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Repeat data along channel dimension if applicable
    n_channels = next(model.parameters()).shape[1]
    if n_channels > 1:
        x = x.repeat(n_channels, axis=-1)

    if orig: 
        loader = ImageLoader(x, device=device, batch_size=batch_size, transform=None, shuffle=False) # transform is None
    else:
        loader = ImageLoader(x, device=device, batch_size=batch_size, transform=SpecAugment(), shuffle=False) #tranform is not None

    with torch.no_grad():
        y_pred = torch.cat([model(batch_x).softmax(dim=1).data
                            for batch_x, in loader])
    
    columns = utils.LABELS[args.class_set_name:args.class_set_name+5]
    return inference.merge_predictions(y_pred.cpu().numpy(), df.index,columns=columns)    

def calc_p_values(n, trial, test_mse, cal_set_mse, is_in_dist_set):

    cal_set_mse_reshaped = cal_set_mse
    cal_set_mse_reshaped = cal_set_mse_reshaped.reshape(1,-1) # cal_set_mse reshaped into row vector

    test_mse_reshaped = test_mse
    test_mse_reshaped = test_mse_reshaped.reshape(-1,1) # test_mse reshaped into column vector

    compare = test_mse_reshaped<=cal_set_mse_reshaped
    p_values = np.sum(compare, axis=1)
    p_values = (p_values+1)/(len(cal_set_mse)+1)
    
    if is_in_dist_set==1:
        np.savez("audio_p_values_n{}_trial{}.npz".format(n,trial), p_values=p_values)
    else:
        np.savez("audio_ood_p_values_n{}_trial{}.npz".format(n,trial), p_values=p_values)

    return p_values

def checkOOD():

    import utils
    import scipy

    # get the data
    x_test, df_test, x_cal, df_cal, x_ood, df_ood = get_orig_data()

    model_path = os.path.join(args.model_path, '{}'.format(args.training_id))

    # standardize data
    mean, std = pickle.load(open(os.path.join(model_path, 'scaler.p'), 'rb'))
    x_test = utils.standardize(x_test, mean, std)
    x_cal = utils.standardize(x_cal, mean, std)
    x_ood = utils.standardize(x_ood, mean, std)
    
    # model predictions on original data
    test_predictions = predict(x=x_test, df=df_test, epoch=args.epochs, model_path=model_path, orig=True)
    cal_predictions = predict(x=x_cal, df=df_cal, epoch=args.epochs, model_path=model_path, orig=True)
    ood_predictions = predict(x=x_ood, df=df_ood, epoch=args.epochs, model_path=model_path, orig=True)

    test_pred_class = np.argmax(test_predictions.to_numpy(),axis=1)
    cal_pred_class = np.argmax(cal_predictions.to_numpy(),axis=1)
    ood_pred_class = np.argmax(ood_predictions.to_numpy(),axis=1)
    
    n = args.n
    auroc_list = []
    tnr_list = []

    baseline_tnr_list = []
    baseline_auroc_list = []
    non_conf_tnr_list = []
    non_conf_auroc_list = []

    for trial in range(args.trials):

        print("Trial no: ", trial+1)

        ood_set_non_conf = []
        in_dist_non_conf = []
        cal_set_non_conf = []


        for iter in range(n):
            # model prediction on transformed data
            trans_test_predictions = predict(x=x_test, df=df_test, epoch=args.epochs, model_path=model_path, orig=False)
            trans_cal_predictions = predict(x=x_cal, df=df_cal, epoch=args.epochs, model_path=model_path, orig=False)
            trans_ood_predictions = predict(x=x_ood, df=df_ood, epoch=args.epochs, model_path=model_path, orig=False)

            trans_test_pred_class = np.argmax(trans_test_predictions.to_numpy(),axis=1)
            trans_cal_pred_class = np.argmax(trans_cal_predictions.to_numpy(),axis=1)
            trans_ood_pred_class = np.argmax(trans_ood_predictions.to_numpy(),axis=1)

            ### non-conformance between orig predictions and transformed predictions
            test_non_conf = np.array((test_pred_class!=trans_test_pred_class),dtype=int)
            in_dist_non_conf.append(test_non_conf)

            cal_non_conf = np.array((cal_pred_class!=trans_cal_pred_class),dtype=int)
            cal_set_non_conf.append(cal_non_conf)

            ood_non_conf = np.array((ood_pred_class!=trans_ood_pred_class),dtype=int)
            ood_set_non_conf.append(ood_non_conf)

            # results with baseline and non-conformance results for OOD detection
            if iter == 0:
                baseline_tnr, baseline_auroc = get_baseline_results(ood_predictions, test_predictions)
                baseline_tnr_list.append(baseline_tnr)
                baseline_auroc_list.append(baseline_auroc)

                non_conf_tnr, non_conf_auroc = get_non_conf_results(ood_non_conf, test_non_conf)
                non_conf_tnr_list.append(non_conf_tnr)
                non_conf_auroc_list.append(non_conf_auroc)

            #pdb.set_trace()
        ########## STEP 1 = for each data point, create n alphas = V(data point) #################
        ood_set_non_conf = np.array(ood_set_non_conf) # ood_set_non_conf = n X |ood set|
        ood_set_non_conf = np.transpose(ood_set_non_conf) # ood_set_non_conf = |ood set| X n
        cal_set_non_conf = np.array(cal_set_non_conf) # cal_set_non_conf = n X |train dataset| 
        cal_set_non_conf = np.transpose(cal_set_non_conf) # cal_set_non_conf = |train dataset| X n
        in_dist_set_non_conf = np.array(in_dist_non_conf) # in_dist_set_non_conf = n X |in_dist_test_dataset|
        in_dist_set_non_conf = np.transpose(in_dist_set_non_conf) # in_dist_set_non_conf = |in_dist_test_dataset| X n

        np.savez("audio_cal_set_non_conf_trial{}.npz".format(trial+1),cal_set_non_conf=cal_set_non_conf) # cal_set_non_conf = 2D array of dim |cal set| X n

        np.savez("audio_in_dist_test_set_non_conf_trial{}.npz".format(trial+1),in_dist_set_non_conf=in_dist_set_non_conf) # in_dist_set_non_conf = 2D array of dim |val set| X n

        np.savez("audio_ood_non_conf_trial{}.npz".format(trial+1),ood_set_non_conf=ood_set_non_conf) # ood_set_non_conf = 2D array of dim |val set| X n

        ######## STEP 2 =  Apply F on V(data point), F(t) = summation of all the values in t 
        f_cal_set = np.sum(cal_set_non_conf, axis = 1)
        f_in_dist_set = np.sum(in_dist_set_non_conf, axis = 1)
        f_ood_set = np.sum(ood_set_non_conf, axis = 1)

        np.savez("audio_in_dist_test_set_summed_non_conf_trial{}.npz".format(trial+1),f_in_dist_set=f_in_dist_set)
        np.savez("audio_ood_set_summed_non_conf_trial{}.npz".format(trial+1), f_ood_set=f_ood_set)

        ######## STEP 3 = Calculate p-values for OOD and validation set #########
        ood_p_values = calc_p_values(n, trial+1, f_ood_set, f_cal_set, is_in_dist_set=0)
        # calculate p-values for test in-dist dataset - higher p-values for in-dist and lower for OODs
        indist_p_values = calc_p_values(n, trial+1, f_in_dist_set, f_cal_set, is_in_dist_set=1)

        indist_p_values = np.sort(indist_p_values)

        epsilon = indist_p_values[int(len(indist_p_values)*0.1)] #OOD detection threshold at 90% TPR

        tnr = np.mean(ood_p_values<epsilon)

        tnr_list.append(tnr*100.)
        
        au_roc =  getAUROC(n, trial+1)

        auroc_list.append(au_roc)

    if (args.n == 20):
        print("SBP (SOTA) results: AUROC {}".format(baseline_auroc_list[0]))
        print("Base Score results: AUROC {} +- {}".format(np.mean(np.array(non_conf_auroc_list)), np.std(np.array(non_conf_auroc_list))))

    # pdb.set_trace()

    return np.mean(np.array(auroc_list)), np.std(np.array(auroc_list)), np.mean(np.array(tnr_list)), np.std(np.array(tnr_list))

def getAUROC(n, trial):
    ood_p_values = np.load("audio_ood_p_values_n{}_trial{}.npz".format(n,trial))['p_values']
    indist_p_values = np.load("audio_p_values_n{}_trial{}.npz".format(n,trial))['p_values']
    p_values = np.concatenate((indist_p_values, ood_p_values))

    # higher p-values for in-dist and lower for OODs
    indist_label = np.ones((len(indist_p_values)))
    ood_label = np.zeros((len(ood_p_values)))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, p_values)*100.
    return au_roc

def get_baseline_results(ood_predictions, test_predictions):

    test_softmax = np.max(test_predictions, axis=1)
    ood_softmax =  np.max(ood_predictions,axis=1)

    test_softmax = np.sort(test_softmax)
    epsilon = test_softmax[int(len(test_softmax)*0.1)] #OOD detection threshold at 90% TPR

    tnr = np.mean(ood_softmax<epsilon)

    in_label = np.ones((len(test_softmax))) # higher softmax scores for in-dist
    out_label = np.zeros((len(ood_softmax)))
    label = np.concatenate((in_label, out_label))
    from sklearn.metrics import roc_auc_score
    softmax = np.concatenate((test_softmax, ood_softmax))
    au_roc = roc_auc_score(label, softmax)*100.

    return tnr*100., au_roc

def get_non_conf_results(ood_non_conf, test_non_conf):

    test_non_conf = np.sort(test_non_conf)
    epsilon = test_non_conf[int(len(test_non_conf)*0.9)] #OOD detection threshold at 90% TPR

    tnr = np.mean(ood_non_conf>epsilon)

    out_label = np.ones((len(ood_non_conf))) # higher non conformance for OODs
    in_label = np.zeros((len(test_non_conf)))
    label = np.concatenate((out_label, in_label))
    from sklearn.metrics import roc_auc_score
    non_conf = np.concatenate((ood_non_conf, test_non_conf))
    au_roc = roc_auc_score(label, non_conf)*100.

    return tnr*100., au_roc

if __name__ == "__main__":
    au_roc_mean, au_roc_std, tnr_mean, tnr_std = checkOOD()
    if args.n == 1:
        print("ICAD results: AUROC: {} +- {}".format(au_roc_mean, au_roc_std))
    else:
        print("Ours with |V(x)| = {} results: AUROC: {} +- {}".format(args.n, au_roc_mean, au_roc_std))
    
