import cv2
from PIL import Image
import argparse
import sys
import os
import torchvision.transforms as trans
import numpy as np
import torch
from config import get_config
from mtcnn import MTCNN
import time
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from utils import load_facebank, prepare_facebank

sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort
sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced")
sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced\retinaface")
from Learner import face_learner
from retinaface.detect_class_hr import FaceDetector
from face_aligner import FaceAligner

def calculate_similarity(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def create_pairs(images):
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append((images[i], images[j]))
    return pairs

def save_aligned_images(pairs, folder_prefix):
    for idx, (face1, face2) in enumerate(pairs):
        pair_folder = f'{folder_prefix}/pair_{idx}'
        if not os.path.exists(pair_folder):
            os.makedirs(pair_folder)
        
        # Convert face1 and face2 to images and save
        if isinstance(face1, list):
            face1 = np.array(face1[0])  # Convert the list to a NumPy array
        if isinstance(face2, list):
            face2 = np.array(face2[0])  # Convert the list to a NumPy array

        # Ensure the images are in the right format
        if face1.dtype != np.uint8:
            face1 = (face1 * 255).astype(np.uint8)
        if face2.dtype != np.uint8:
            face2 = (face2 * 255).astype(np.uint8)

        Image.fromarray(face1).save(f'{pair_folder}/Alignedimg1.png')
        Image.fromarray(face2).save(f'{pair_folder}/Alignedimg2.png')

def rgb_to_bgr(img):
    # Ensure img is a PIL Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    # Convert the PIL Image to a PyTorch tensor
    img_tensor = trans.ToTensor()(img)
    # Convert RGB to BGR by flipping the channels
    bgr_img_tensor = torch.flip(img_tensor, dims=[0])
    # Convert the PyTorch tensor back to a PIL Image
    bgr_img = trans.ToPILImage()(bgr_img_tensor)
    return bgr_img

def get_embedding(learner, conf, image):
    resize_dim = (256, 256)  # Example dimension, adjust as needed

    if isinstance(image, Image.Image):
        image = np.array(image)
    resized_image = Image.fromarray(image).resize(resize_dim, Image.BILINEAR)
    
    # Align the resized image
    faces, _, _ = rface.align_multi(np.array(resized_image))
    if faces is None or len(faces) == 0:
        return None
    
    # Take the first face from the list
    face = faces[0]
    if isinstance(face, torch.Tensor):
        face = face.permute(1, 2, 0).cpu().numpy()
    face = Image.fromarray(face)

    required_size = (112, 112)
    face = face.resize(required_size, Image.BILINEAR)

    bgr_face = rgb_to_bgr(face)

    face_tensor = conf.test_transform(bgr_face).to(conf.device).unsqueeze(0)
    with torch.no_grad():
        embedding = learner.model(face_tensor)
    return embedding.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name", default='videomy.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    parser.add_argument("-u", "--update", help="whether to perform update the facebank", action="store_true")
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether to show the confidence score", action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection (in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long (in seconds)", default=0, type=int)
    parser.add_argument("--bbox_threshold", help="threshold for bounding box size", default=100, type=int)
    args = parser.parse_args()
    
    conf = get_config(False)
    mtcnn = MTCNN()
    print('MTCNN loaded')
    detector = FaceDetector()
    print('FaceDetector loaded')
    rface = FaceAligner(detector.detect_landmarks, desiredFaceWidth=112, desiredFaceHeight=112)
    print('FaceAligner loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\work_space\save\models.pth', True, True)
    else:
        learner.load_state(conf, r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\model_ir_se50.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:   
        if conf.detector_name == 'retinaface':
            targets, names = prepare_facebank(conf, learner.model, rface, tta=args.tta)
        elif conf.detector_name == 'mtcnn':
            targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
        else:
            print('detector not found for facebank update')
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # Paths to genuine and impostor folders
    genuine_folder = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\guninevedios'
    impostor_folder = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\postervedio'

    # Load images from folders
    genuine_images = load_images_from_folder(genuine_folder)
    impostor_images = load_images_from_folder(impostor_folder)

    # Create genuine pairs (same person)
    genuine_pairs = create_pairs(genuine_images)
    # Create impostor pairs (different people)
    impostor_pairs = []
    for genuine_image in genuine_images:
        for impostor_image in impostor_images:
            impostor_pairs.append((genuine_image, impostor_image))

    # Save aligned images
    save_aligned_images(genuine_pairs, 'genuine')
    save_aligned_images(impostor_pairs, 'impostor')

    # # Printing pairs for verification
    # print("Genuine pairs:")
    # for pair in genuine_pairs:
    #     print(pair)
    # print("Genuine pairs completed")

    # print("Impostor pairs completed")
    print("Length of genuine pairs:", len(genuine_pairs))
    print("Length of impostor pairs:", len(impostor_pairs))

    genuine_embeddings = [(get_embedding(learner, conf, img1), get_embedding(learner, conf, img2)) for img1, img2 in genuine_pairs]
    impostor_embeddings = [(get_embedding(learner, conf, img1), get_embedding(learner, conf, img2)) for img1, img2 in impostor_pairs]
    print("Embeddings are calculated")

    genuine_scores = [calculate_similarity(emb1, emb2) for emb1, emb2 in genuine_embeddings if emb1 is not None and emb2 is not None]
    impostor_scores = [calculate_similarity(emb1, emb2) for emb1, emb2 in impostor_embeddings if emb1 is not None and emb2 is not None]
    print("Scores calculated")

    genuine_labels = [1] * len(genuine_scores)
    impostor_labels = [0] * len(impostor_scores)
    print("Labels assigned")

    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([genuine_labels, impostor_labels])
    print("Scores concatenated")

    # Generate at least 30 thresholds
    thresholds = np.linspace(min(scores), max(scores), num=30)
    
    fpr = []
    tpr = []
    for threshold in thresholds:
        tp = np.sum((scores <= threshold) & (labels == 1))
        fp = np.sum((scores <= threshold) & (labels == 0))
        tn = np.sum((scores > threshold) & (labels == 0))
        fn = np.sum((scores > threshold) & (labels == 1))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    roc_auc = auc(fpr, tpr)

 # After calculating fpr, tpr, and scores

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_curve.png')  # Save the ROC curve plot
plt.show()

print("ROC AUC score:", roc_auc)

# Plot histogram
plt.figure()
plt.hist(genuine_scores, bins=30, alpha=0.5, label='Genuine')
plt.hist(impostor_scores, bins=30, alpha=0.5, label='Impostor')
plt.title('Histogram of Similarity Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('score_histograms.png')  # Save the histogram plot
plt.show()






#****************************** without alingmnt*******************************
# import cv2
# from PIL import Image
# import argparse
# import sys
# import os
# import torchvision.transforms as trans
# import numpy as np
# import torch
# from config import get_config
# from mtcnn import MTCNN
# import time
# from sklearn.metrics import auc
# import matplotlib.pyplot as plt
# from utils import load_facebank, prepare_facebank

# sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
# from sort import Sort
# sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced")
# sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced\retinaface")
# from Learner import face_learner
# from retinaface.detect_class_hr import FaceDetector
# from face_aligner import FaceAligner

# def calculate_similarity(embedding1, embedding2):
#     return np.linalg.norm(embedding1 - embedding2)

# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img_path = os.path.join(folder, filename)
#         img = cv2.imread(img_path)
#         if img is not None:
#             images.append(img)
#     return images

# def create_pairs(images):
#     pairs = []
#     for i in range(len(images)):
#         for j in range(i + 1, len(images)):
#             pairs.append((images[i], images[j]))
#     return pairs

# def save_aligned_images(pairs, folder_prefix):
#     if not os.path.exists(folder_prefix):
#         os.makedirs(folder_prefix)
        
#     for idx, (face1, face2) in enumerate(pairs):
#         # Convert face1 and face2 to images and save
#         if isinstance(face1, list):
#             face1 = np.array(face1[0])  # Convert the list to a NumPy array
#         if isinstance(face2, list):
#             face2 = np.array(face2[0])  # Convert the list to a NumPy array

#         # Ensure the images are in the right format
#         if face1.dtype != np.uint8:
#             face1 = (face1 * 255).astype(np.uint8)
#         if face2.dtype != np.uint8:
#             face2 = (face2 * 255).astype(np.uint8)

#         Image.fromarray(face1).save(f'{folder_prefix}/aligned_face1_{idx}.png')
#         Image.fromarray(face2).save(f'{folder_prefix}/aligned_face2_{idx}.png')

# def rgb_to_bgr(img):
#     # Ensure img is a PIL Image
#     if not isinstance(img, Image.Image):
#         img = Image.fromarray(img)
#     # Convert the PIL Image to a PyTorch tensor
#     img_tensor = trans.ToTensor()(img)
#     # Convert RGB to BGR by flipping the channels
#     bgr_img_tensor = torch.flip(img_tensor, dims=[0])
#     # Convert the PyTorch tensor back to a PIL Image
#     bgr_img = trans.ToPILImage()(bgr_img_tensor)
#     return bgr_img

# def get_embedding(learner, conf, image):
#     resize_dim = (256, 256)  # Example dimension, adjust as needed

#     if isinstance(image, Image.Image):
#         image = np.array(image)
        
#     face = face.resize(resize_dim, Image.BILINEAR)

#     # resized_image = Image.fromarray(image).resize(resize_dim, Image.BILINEAR)
    
#     # # Align the resized image
#     # faces, _, _ = rface.align_multi(np.array(resized_image))
#     # if faces is None or len(faces) == 0:
#     #     return None
    
#     # Take the first face from the list
    
#     if isinstance(face, torch.Tensor):
#         face = face.permute(1, 2, 0).cpu().numpy()
#     face = Image.fromarray(face)

#     required_size = (112, 112)
#     face = face.resize(required_size, Image.BILINEAR)

#     bgr_face = rgb_to_bgr(face)

#     face_tensor = conf.test_transform(bgr_face).to(conf.device).unsqueeze(0)
#     with torch.no_grad():
#         embedding = learner.model(face_tensor)
#     return embedding.cpu().numpy()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='for face verification')
#     parser.add_argument("-f", "--file_name", help="video file name", default='videomy.mp4', type=str)
#     parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
#     parser.add_argument("-u", "--update", help="whether to perform update the facebank", action="store_true")
#     parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
#     parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
#     parser.add_argument("-c", "--score", help="whether to show the confidence score", action="store_true")
#     parser.add_argument("-b", "--begin", help="from when to start detection (in seconds)", default=0, type=int)
#     parser.add_argument("-d", "--duration", help="perform detection for how long (in seconds)", default=0, type=int)
#     parser.add_argument("--bbox_threshold", help="threshold for bounding box size", default=100, type=int)
#     args = parser.parse_args()
    
#     conf = get_config(False)
#     mtcnn = MTCNN()
#     print('MTCNN loaded')
#     detector = FaceDetector()
#     print('FaceDetector loaded')
#     rface = FaceAligner(detector.detect_landmarks, desiredFaceWidth=112, desiredFaceHeight=112)
#     print('FaceAligner loaded')
    
#     learner = face_learner(conf, True)
#     learner.threshold = args.threshold
#     if conf.device.type == 'cpu':
#         learner.load_state(conf, r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\work_space\save\models.pth', True, True)
#     else:
#         learner.load_state(conf, r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\model_ir_se50.pth', True, True)
#     learner.model.eval()
#     print('learner loaded')
    
#     if args.update:
#         if conf.detector_name == 'retinaface':
#             targets, names = prepare_facebank(conf, learner.model, rface, tta=args.tta)
#         elif conf.detector_name == 'mtcnn':
#             targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
#         else:
#             print('detector not found for facebank update')
#         print('facebank updated')
#     else:
#         targets, names = load_facebank(conf)
#         print('facebank loaded')

#     # Paths to genuine and impostor folders
#     genuine_folder = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\guninevedios'
#     impostor_folder = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\postervedio'

#     # Load images from folders
#     genuine_images = load_images_from_folder(genuine_folder)
#     impostor_images = load_images_from_folder(impostor_folder)

#     # Create genuine pairs (same person)
#     genuine_pairs = create_pairs(genuine_images)
#     # Create impostor pairs (different people)
#     impostor_pairs = []
#     for genuine_image in genuine_images:
#         for impostor_image in impostor_images:
#             impostor_pairs.append((genuine_image, impostor_image))

#     # Save aligned images
#     save_aligned_images(genuine_pairs, 'genuine')
#     save_aligned_images(impostor_pairs, 'impostor')

#     # Printing pairs for verification
#     print("Genuine pairs:")
#     for pair in genuine_pairs:
#         print(pair)
#     print("Genuine pairs completed")

#     print("Impostor pairs completed")
#     print("Length of genuine pairs:", len(genuine_pairs))
#     print("Length of impostor pairs:", len(impostor_pairs))

#     genuine_embeddings = [(get_embedding(learner, conf, img1), get_embedding(learner, conf, img2)) for img1, img2 in genuine_pairs]
#     impostor_embeddings = [(get_embedding(learner, conf, img1), get_embedding(learner, conf, img2)) for img1, img2 in impostor_pairs]
#     print("Embeddings are calculated")

#     genuine_scores = [calculate_similarity(emb1, emb2) for emb1, emb2 in genuine_embeddings if emb1 is not None and emb2 is not None]
#     impostor_scores = [calculate_similarity(emb1, emb2) for emb1, emb2 in impostor_embeddings if emb1 is not None and emb2 is not None]
#     print("Scores calculated")

#     genuine_labels = [1] * len(genuine_scores)
#     impostor_labels = [0] * len(impostor_scores)
#     print("Labels assigned")

#     scores = np.concatenate([genuine_scores, impostor_scores])
#     labels = np.concatenate([genuine_labels, impostor_labels])
#     print("Scores concatenated")

#     # Generate at least 30 thresholds
#     thresholds = np.linspace(min(scores), max(scores), num=30)
    
#     fpr = []
#     tpr = []
#     for threshold in thresholds:
#         tp = np.sum((scores <= threshold) & (labels == 1))
#         fp = np.sum((scores <= threshold) & (labels == 0))
#         fn = np.sum((scores > threshold) & (labels == 1))
#         tn = np.sum((scores > threshold) & (labels == 0))
        
#         fpr.append(fp / (fp + tn))
#         tpr.append(tp / (tp + fn))

#     fpr = np.array(fpr)
#     tpr = np.array(tpr)

#     print("FPR:", fpr)
#     print("TPR:", tpr)
#     print("Thresholds:", thresholds)

#     roc_auc = auc(fpr, tpr)
#     print("ROC AUC:", roc_auc)
 
#     plt.figure()
#     plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC)')
#     plt.legend(loc='lower right')
#     plt.savefig('roc_curve.png')
#     plt.show()
#     print("ROC curve plotted and saved")

#     plt.figure()
#     plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine')
#     plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor')
#     plt.xlabel('Similarity Score')
#     plt.ylabel('Frequency')
#     plt.title('Score Histograms')
#     plt.legend(loc='upper right')
#     plt.savefig('score_histograms.png')
#     plt.show()
#     print("Score histograms plotted and saved")
