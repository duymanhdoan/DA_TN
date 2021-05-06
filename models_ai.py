import cv2
import numpy as np 
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import args 
import matplotlib.pyplot as plt
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from torchvision import transforms as trans

class FaceRecognize(object): 
    def __init__(self, args): 
        # --- init mtcnn face detection models ------ 
        self.mtcnn = MTCNN()

        # ---- init models ----- 
        if args.model_type == 'mobile_facenet': 
            self.model = MobileFaceNet(args.embedding_size).to(args.device)
            print('{}_{} model generated'.format(args.net_mode, args.net_depth))
        elif args.model_type == 'resnet': 
            self.model = Backbone(args.net_depth, args.drop_ratio, args.net_mode).to(args.device)
            print('{}_{} model generated'.format(args.net_mode, args.net_depth))
        else: 
            raise TypeError("does not support for model type {}".format(args.model_type))
        
        # ---- load weight ----- 
        self.model.load_state_dict(torch.load(args.pretrained_path,  map_location = args.device))
        print('load models done!\n')
        self.model.eval()

        #----- get threshold ------
        self.threshold = args.threshold

    def update_facedb(self, args):
        # ----- update database ------ 
        if args.update:
            targets, names = prepare_facebank(args, self.model, self.mtcnn, tta = args.tta)
            print('facebank updated')
        else:
            targets, names = load_facebank(args)
            print('facebank loaded')
        return targets, names
  

    def infer(self, args, faces, target_embs, tta=False):
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
                emb = self.model(args.test_transform(img).to(args.device).unsqueeze(0))
                emb_mirror = self.model(args.test_transform(mirror).to(args.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(args.test_transform(img).to(args.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum 

    def face_recognize(self, frame, args, targets, names):
        try:
        # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            image = Image.fromarray(frame)
            bboxes, faces = self.mtcnn.align_multi(image, args.face_limit, args.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice
            results, score = self.infer(args, faces, targets, args.tta)
            for idx,bbox in enumerate(bboxes):
                bbox = np.array(bbox,dtype=int)
                if args.score:
                    frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                else:
                    frame = draw_box_name(bbox, names[results[idx] + 1], frame)
        except:
            print('detect error')

        return frame


# if __name__ == '__main__':

#     folder_image = '/home/minglee/Documents/DA_TN/trash'
#     mtcnn = MTCNN()
#     print('mtcnn loaded')
    
#     learner = InferenceModels(args)

#     learner.model.eval()
#     print('models learner eval status')
    
#     if args.update:
#         targets, names = prepare_facebank(args, learner.model, mtcnn, tta = args.tta)
#         print('facebank updated')
#     else:
#         targets, names = load_facebank(args)
#         print('facebank loaded')

#     # inital camera
#     cap = cv2.VideoCapture(0)
#     cap.set(3,1280)
#     cap.set(4,720)
   
#     while cap.isOpened():
#         isSuccess,frame = cap.read()
#         if isSuccess:            
#             try:
#                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
#                 # image = Image.fromarray(frame)
#                 bboxes, faces = mtcnn.align_multi(image, args.face_limit, args.min_face_size)
#                 bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
#                 bboxes = bboxes.astype(int)
#                 bboxes = bboxes + [-1,-1,1,1] # personal choice
#                 results, score = learner.infer(args, faces, targets, args.tta)
#                 for idx,bbox in enumerate(bboxes):
#                     bbox = np.array(bbox,dtype=int)
#                     print(bbox)
#                     if args.score:
#                         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
#                     else:
#                         frame = draw_box_name(bbox, names[results[idx] + 1], frame)
#             except:
#                 print('detect error') 
#             cv2.imshow('face Capture', frame)

#         if cv2.waitKey(1)&0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()    