
import adafacenet
from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate
import torch
from torch import optim 
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import os

class face_learner(object):
    def __init__(self, conf, inference=False):
        # print(conf)
        print(f'verifying model name before passing it to the conditional block', conf.model_name)
        if conf.model_name == 'treb_mobile':
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        elif conf.model_name == 'treb_resnet' or conf.model_name == 'our_model':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)        
            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
            print('two model heads generated')
            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold
            # print("else block execution.....")

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):

        
        adaface_models = {
                'ir_50':r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\work_space\save\adaface_ir50_webface4m.ckpt',
                }
            



            

        def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
            assert architecture in adaface_models.keys()
            model = adafacenet.build_model(architecture)

            checkpoint_path = adaface_models[architecture]
    
            # Print the full path being accessed
            print(f"Attempting to load checkpoint from: {os.path.abspath(checkpoint_path)}")
            statedict = torch.load(adaface_models[architecture])['state_dict']
            model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
            model.load_state_dict(model_statedict)
            model.eval()
            return model
        
        if from_save_folder:
            save_path = conf.save_path
            print('from save folder', conf.save_path)
        else:
            save_path = conf.model_path
            print(conf.model_path)

        # Determine the model filename based on the model name
        if conf.model_name == 'treb_mobile':
            model_filename = 'model_mobilefacenet.pth'
        elif  conf.model_name == 'treb_resnet':  
              model_filename = 'model_ir_se50.pth'

        elif conf.model_name == 'our_model': 
              model_filename = 'model_for_Nadeem.pth'#'model_for_Nadeem.pth' , model_mohsin_full_version
              print("our model being used")
        elif conf.model_name=='Adaface':

            print('adaface model is  being tried to load ')

            # adaface_models = {
            #     'ir_50':r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\work_space\save\adaface_ir50_webface4m.ckpt',
            #     }
            
        else:
                print('no model found ')

        # if conf.model_name=='our_model': #trebresnet
        if conf.model_name in ['treb_resnet', 'treb_mobile', 'our_model']:

        # Load the model state dictionary
           self.checkpoint  = torch.load(os.path.join(save_path, model_filename), map_location=torch.device('cpu'))
            
        #    print("Checkpoint keys from treb..........:", self.checkpoint.keys())

           self.model.load_state_dict(torch.load(os.path.join(save_path, model_filename), map_location=torch.device('cpu')))

           if not model_only:
                self.head.load_state_dict(torch.load(os.path.join(save_path, 'head_{}'.format(fixed_str))))
                self.optimizer.load_state_dict(torch.load(os.path.join(save_path, 'optimizer_{}'.format(fixed_str))))

        # elif  conf.model_name == 'our_model': 
        #     self. model = Backbone(50, 0.5, 'ir_se')
            
        #     # Define the path to your model file
        #     model_path = os.path.join(save_path, model_filename)

        #     # Load the model with consideration of CUDA availability
        #     if torch.cuda.is_available():
        #        self.checkpoint = torch.load(model_path)
        #     else:
        #         self.checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        #         self.model.load_state_dict(torch.load(os.path.join(save_path, model_filename), map_location=torch.device('cpu')))

        #     # print("Checkpoint keys:", self.checkpoint.keys())
        #     print("Checkpoint keys from mdele   :", model_filename)
        else:
          self. model=load_pretrained_model('ir_50')
        #   print("Checkpoint keys from adaface..........:", adaface_models.keys())





    


    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
    
    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def find_lr(self, conf, init_value=1e-8, final_value=10., beta=0.98, bloding_scale=3., num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):
            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          
            self.optimizer.zero_grad()
            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)
            avg_loss = beta * avg_loss + (1 - beta) *loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                break
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            loss.backward()
            self.optimizer.step()
            for params in self.optimizer.param_groups:
                params['lr'] *= mult
            lr *= mult
        return log_lrs, losses





















    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.            
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()                                 
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                    
                self.step += 1
                
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    # ... (previous code) 


    # ... (previous code)

    # def infer(self, conf, faces, target_embs, tta=False):
    #     '''
    #     faces : list of PIL Image
    #     target_embs : [n, 512] computed embeddings of faces in facebank
    #     names : recorded names of faces in facebank
    #     tta : test time augmentation (hfilp, that's all)
    #     '''
    #     embs = []
    #     for img in faces:
    #         if tta:
    #             mirror = trans.functional.hflip(img)
    #             emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
    #             emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
    #             embs.append(l2_norm(emb + emb_mirror))
    #         else:                        
    #             embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    #     source_embs = torch.cat(embs)
    #     target_embs_tensor = torch.cat(target_embs, dim=0)  # Concatenate target embeddings as a tensor
    #     diff = source_embs.unsqueeze(-1) - target_embs_tensor.transpose(1, 0).unsqueeze(0)

    #     dist = torch.sum(torch.pow(diff, 2), dim=1)
    #     minimum, min_idx = torch.min(dist, dim=1)

    #     print("Source embeddings: ", source_embs)
    #     print(  "Target embeddings: ", target_embs_tensor)

    #     print(f"Source embeddings shape: {source_embs.shape}")
    #     print(f"Target embeddings shape: {target_embs_tensor.shape}")
    #     # print (f'size of  embeding ==  {len(source_embs)} and size of terget emb= {len(target_embs_tensor)} ' )



    #     min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
    #     return min_idx, minimum







   
    def infer(self, conf, faces, target_embs, tta=False):

        def rgb_to_bgr(img):
             # Convert the PIL Image to a PyTorch tensor
            img_tensor = trans.ToTensor()(img)
            # Convert RGB to BGR by flipping the channels
            bgr_img_tensor = torch.flip(img_tensor, dims=[0])
            # Convert the PyTorch tensor back to a PIL Image
            bgr_img = trans.ToPILImage()(bgr_img_tensor)
            return bgr_img
        #     '''
        embs = []
        for img in faces:
            # Convert RGB image to BGR format
            bgr_img = rgb_to_bgr(img)
            if tta:
                mirror = trans.functional.hflip(bgr_img)
                emb = self.model(conf.test_transform(bgr_img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(bgr_img).to(conf.device).unsqueeze(0)))

        source_embs = torch.cat(embs)
        target_embs_tensor = torch.cat(target_embs, dim=0)  # Concatenate target embeddings as a tensor
        diff = source_embs.unsqueeze(-1) - target_embs_tensor.transpose(1, 0).unsqueeze(0)

        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)

        # print("Source embeddings: ", source_embs)
        # print("Target embeddings: ", target_embs_tensor)

        # print(f"Source embeddings shape: {source_embs.shape}")
        # print(f"Target embeddings shape: {target_embs_tensor.shape}")

        min_idx[minimum > conf.threshold ] = -1  # if no match, set idx to -1
        return min_idx, minimum
