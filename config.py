from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config():
    conf = edict()
    conf.data_path = Path('../TTTN/embedding')
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = True
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.score = True
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])


    conf.pretrain_paths = Path('../TTTN/pretrained/model_mobilefacenet.pth')
    conf.tta = True
    conf.update = True
   
    conf.facebank_path = conf.data_path/'facebank'
    conf.threshold = 1.0
    conf.face_limit = 10
    #when inference, at maximum detect 10 faces in one image, my laptop is slow
    conf.min_face_size = 30
    # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
