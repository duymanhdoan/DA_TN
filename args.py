from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans




data_root = Path('/home/duydm/Documents/InsightFace_Pytorch')
work_path = Path('output_models/')
model_path = work_path/'models'
log_path = work_path/'log'
save_path = work_path/'save'
input_size = [112,112]
embedding_size = 512
use_mobilfacenet = False
net_depth = 50
drop_ratio = 0.6
net_mode = 'ir_se' # or 'ir'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
#--------------------Training Config ------------------------   
data_mode = 'emore'
emore_folder = data_root/'faces_emore'
batch_size = 8 # irse net depth 50    
log_path = work_path/'tensorboard'
save_path = work_path/'save'
epochs = 5
num_workers = 3
ce_loss = CrossEntropyLoss()    
    #     conf.weight_decay = 5e-4

lr = 1e-3
milestones = [12,15,18]
momentum = 0.9
pin_memory = True

#--------------------Inference Config ------------------------

acebank_path = work_path/'facebank'
threshold = 1.5
face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
#________________________ evaluate dataset _________________________________ 
eval_dataset = Path('/home/duydm/Documents/InsightFace_Pytorch/dataset')
list_name_eval = ['agedb_30']
# list_name_eval = [ 'lfw', 'cplfw' ,'cfp_fp' ,'cfp_ff' ,'calfw' ,'agedb_30','vgg2_fp']

result_eval_file = '/home/duydm/Documents/InsightFace_Pytorch/eval_file.txt'