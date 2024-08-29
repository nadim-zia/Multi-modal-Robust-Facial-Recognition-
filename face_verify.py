import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config , cfg_mnet, cfg_re50

from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


import sys
sys.path.append(r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retina face\Pytorch_Retinaface-master\models')

from net import MobileNetV1 as MobileNetV1
from net import FPN as FPN
from net import SSH as SSH
from retinaface import RetinaFace as Rf

  



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    rf=Rf(cfg=cfg_mnet, phase='test')
    print('mtcnn loaded')
    print('Retina_face loaded')

    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
        
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    args.update = True
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        # targets, names = load_facebank(conf)
        # print('facebank loaded')
        print ('ya bakwas hoaraha ha' )

    # inital camera
    # cap = cv2.VideoCapture(0)
    # cap.set(3,1280)
    # cap.set(4,720)
    # open the video file
    video_path = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data/vediomy.mp4'
    cap = cv2.VideoCapture(video_path)
    print("vedio run succesfully")

# check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

    if args.save:
        video_writer = cv2.VideoWriter(conf.data_path/'recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (1280,720))
        # frame rate 6 due to my laptop is quite slow...
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:             
            
            try:
#                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice    
                results, score = learner.infer(conf, faces, targets, args.tta)
                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except:
                print('detect error')    
                
            cv2.imshow('face Capture', frame)

        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()    