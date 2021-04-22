from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import os

class face_learner(object):
    def __init__(self, conf, inference=False):

        self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
        print('MobileFaceNet model generated')
        if conf.pretrain_paths.exists():
            self.model.load_state_dict(torch.load(conf.pretrain_paths,map_location= conf.device ))    
            print('load pretrained done! ')   
        self.threshold = conf.threshold

    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        # print("distance: {}".format(dist))
        minimum, min_idx = torch.min(dist, dim=1)
        # print(minimum)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum
