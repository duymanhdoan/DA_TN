from data.data_pipe import de_preprocess, get_train_loader, get_val_data
        
import args 
import cv2
import bcolz
import pickle
import torch
import numpy as np 
import mxnet as mx
from tqdm import tqdm
from data.data_pipe import get_val_pair
from torchvision import transforms as trans
import time
from PIL import Image
from tqdm import tqdm
from verifacation import evaluate
from prettytable import PrettyTable
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm



def eval(conf, model, writer, step, epochs):
    
    pretty_tabel = PrettyTable([ 'accuracy', 'best threshold', 'data_type'])
    for name in conf.list_name_eval: 
        print('evaluate dataset type:{}!\n'.format(name))
        carry, issame = get_val_pair(conf.eval_dataset, name)
        carry = carry[:50]
        issame = issame[:50]
        accuracy, best_threshold, tpr, fpr = embedding_ex(conf, carry, issame, model , nrof_folds=10, tta=True)
        pretty_tabel.add_row((accuracy, best_threshold, name ))
        writer.add_scalar('Accuracy_{}_dataset'.format(name), accuracy, step)
        writer.add_scalar('Best_threshold_{}_dataset'.format(name), best_threshold, step)
        # writer.add_scalar('True positive_rate_{}_dataset'.format(name), tpr, step)
        # writer.add_scalar('False_positive_rate_{}_dataset'.format(name), fpr, step)
        
    print(pretty_tabel)
    f = open(conf.result_eval_file, "a") 
    for row in pretty_tabel: 
        f.write(row.get_string())
    f.close()
    f.write('\n')

def embedding_ex(conf, carray, issame, model, nrof_folds = 5, tta = False):
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), conf.embedding_size])
    with torch.no_grad():
        while idx + conf.batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + conf.batch_size])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(conf.device)) + model(fliped.to(conf.device))
                embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
            else:
                embeddings[idx:idx + conf.batch_size] = model(batch.to(conf.device)).cpu()
            idx += conf.batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])            
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(conf.device)) + model(fliped.to(conf.device))
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                embeddings[idx:] = model(batch.to(conf.device)).cpu()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean(), tpr, fpr



if __name__ == '__main__': 
    path = args.eval_dataset
    name = 'agedb_30'
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    carray = carray[:50]
    issame = issame[:50]
    print(carray.shape)
    print(issame.shape)

