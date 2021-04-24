import torch 

# _____________________________invironment training (cuda:0 , cpu)_________________________ 
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# metric 
distance_metric = 0 # 0 is euclidian distance \ 1 is consin similarity. 


#_________________ defind backbone __________________ 
backbone_type =  'ResNet'    # ['ir', 'ir_se'], 'mode should be ir or ir_se' ,[50, 100, 152], 'num_layers should be 50,100, or 152'
loss_type = 'ArcFace'   # help = "Mobilefacenets, Resnet."   support for type loss "mv-softmax, arcface, npc-face."


# ________________ training _____________________________

data_root = '/media/minhdc/DATA/duydmFabbi/dataFace/faces_emore/imgs'
#data_root = '/home/duydm/CodeProject/DA_TN/pretrained/VN_celeb'
train_file = '/home/duydm/CodeProject/DA_TN/train.txt'
batch_size = 64 # help evaluate dataset 
epoches = 30  # number of epoch for training 
step = '10, 13, 16'  # help = 'Step for schedule lr.'
print_freq = 5  # help = 'The print frequency for training state.'
save_freq = 1000  # help = 'The save frequency for training state.'
eval_by_batch_idx = 1000    # number of step evaluate dataset 
reload_model = True # help = 'Whether to resume from a checkpoint. load status model '
num_class = 72778  #number of class
feat_dim = 512 #shape of embedding
image_shape = (112,112) # shape of image 
num_workers = 4 #number of workers 
momentum = 0.9  # help = 'The momentum for sgd.'
status_eval = False
lr = 0.1  # help='The initial learning rate.'
# ___________________ evaluate dataset _______________________

num_of_pair = 120000

root_eval_dataset = '/home/duydm/CodeProject/DA_TN/pretrained/VN_celeb'  # data without mask 
evaluate_batch_size = 64  # batch size of evaluate

# ______________________ work place output model _____________________________
out_dir = 'Output_models'  # help = "The place of folder to save models log history training"
log_dir = 'Output_models/history/'  # help = 'The directory to save log.log'
tensorboardx_logdir = 'tensorboard'  # help = 'The directory to save tensorboardx logs'
#help = "the log file in training or testing "
result_test_file = '/home/duydm/CodeProject/DA_TN/result.txt'
#________________________ The path of pretrained model ____________________________________
pretrain_model = '/home/duydm/CodeProject/DA_TN/pretrained/resnet50ir/Epoch_17.pt' # resnet ir-se 152

# ______________________ define parametor of backbone and loss type __________________________________
model_parameter = {'ResNet': 
                      {'depth': 50,   # 50,100, or 152'
                      'drop_ratio': 0.4, 
                      'net_mode': 'ir',  # ['ir', 'ir_se']
                      'feat_dim': feat_dim, 
                      'out_h': 7, 
                      'out_w': 7},

                     'MobileFaceNet': 
                      {'feat_dim': feat_dim, 
                      'out_h': 7, 
                      'out_w': 7 }
                      } 
# model type   

loss_parameter = {'ArcFace':
                        {'feat_dim': feat_dim,
                        'num_class': num_class,
                        'margin_arc': 0.35,
                        'margin_am': 0.0,
                        'scale': 32},
                    'AM-Softmax':
                        {'feat_dim': feat_dim,
                        'num_class': num_class,
                        'margin': 0.35,
                        'scale': 32}
                    }
# loss type 
