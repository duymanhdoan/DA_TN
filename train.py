from Learner import face_learner
import argparse
import args 
# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    learner = face_learner(args)
    learner.train(args)