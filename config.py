from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans
# python infer_on_video.py --update
#  python infer_on_video.py --update 
def get_config(training = True):
    conf = edict()
    print(454)

    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112] 
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.model_name = 'Adaface'# treb_resnet,treb_mobile,Adaface,our_model
    conf.detector_name = 'retinaface'# mtcnn and retinaface
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.architecture='resnet18'# resnet50
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100 # irse net depth 50  
    # conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True 
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 4
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------

    else:
        
        conf.facebank_path = Path(r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\facebank')
        conf.threshold = 1.54
        conf.face_limit = 50
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces

    return conf

