import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import args



def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)



if __name__ == '__main__':

    conf = get_config()

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)

    learner.model.eval()

    if conf.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280.0)
    cap.set(4, 720.0)
    print("get with: {} and heights: {} of windown".format(cap.get(3),cap.get(4)))

    # if args.save:
    #     video_writer = cv2.VideoWriter(conf.data_path/'recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:
            try:
                # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                # print('len of bboxes: {} \n'.format(len(bboxes)))
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice
                results, score = learner.infer(conf, faces, targets, conf.tta)
                # print('results: {} -> score: {}'.format(results,score))
                for idx,bbox in enumerate(bboxes):
                    if conf.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except:
                print('detect error')

            frame150 = rescale_frame(frame, percent=200)
            cv2.imshow('face Capture', frame150)

        # if args.save:
        #     video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
    # if args.save:
    #     video_writer.release()
    cv2.destroyAllWindows()
