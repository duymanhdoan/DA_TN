from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
#---------- device -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
# ---------- inferences ----------------
action="store_true"
default=1.54
save = False
score = True
data_path = Path('/home/minglee/Documents/DA_TN/database')
facebank_path = data_path/'facebank'
face_limit = 10 
#when inference, at maximum detect 10 faces in one image, my laptop is slow
min_face_size = 30
pnet_threshold = 0.1
model_type = 'mobile_facenet'

<<<<<<< HEAD

data_root = Path('/home/duydm/Documents/DA_TN/place_models')
work_path = Path('/home/duydm/Documents/DA_TN/place_models/Output_models')
=======
#___ train models
data_root = Path('/home/minglee/Documents/DA_TN/place_models')
work_path = Path('/home/minglee/Documents/DA_TN/place_models/output_models/')
>>>>>>> 049ff15ec809681ba617abc69d3ef947a409f3f6
model_path = work_path/'models'
log_path = work_path/'tensorboard'
save_model_path = work_path/'save_pretrained'
input_size = [112,112]
embedding_size = 512
use_mobilfacenet = False
net_depth = 50
drop_ratio = 0.6
net_mode = 'ir_se' # or 'ir'
update = True
tta = True
test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
#--------------------Training Config ------------------------   
data_mode = 'emore'
emore_folder = data_root/'faces_emore'
batch_size = 4 # irse net depth 50    
epochs = 5
num_workers = 3
ce_loss = CrossEntropyLoss()    
lr = 1e-3
resume = True
add_scalar_freq = 1 
save_checkpoint = 10000 
eval_every_checkpoint = 1
milestones = [12,15,18]
momentum = 0.9
pin_memory = True
<<<<<<< HEAD
pretrained_path = '/home/duydm/Documents/DA_TN/place_models/pretrained/insightface_pretrain/model_ir_se50.pth'
=======
pretrained_path = '/home/minglee/Documents/DA_TN/place_models/pretrained/model_mobilefacenet.pth'
>>>>>>> 049ff15ec809681ba617abc69d3ef947a409f3f6
#--------------------Inference Config ------------------------

acebank_path = work_path/'facebank'
threshold = 1.0
face_limit = 5 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
#________________________ evaluate dataset _________________________________ 
eval_dataset = Path('/home/duydm/Documents/DA_TN/place_models/dataset')
list_name_eval = ['agedb_30']
#list_name_eval = [ 'lfw', 'cplfw' ,'cfp_fp' ,'cfp_ff' ,'calfw' ,'agedb_30','vgg2_fp']

<<<<<<< HEAD
result_eval_file = '/home/duydm/Documents/DA_TN/place_models/result_eval.txt'
=======
result_eval_file = '/home/minglee/Documents/DA_TN/place_models/result_eval.txt'


>>>>>>> 049ff15ec809681ba617abc69d3ef947a409f3f6
