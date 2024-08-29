# # Import necessary modules
# import cv2
# from PIL import Image
# import argparse
# from pathlib import Path
# import sys
# import os
# import numpy as np
# import pickle    
# import torch
# import time
# from config import get_config
# from mtcnn import MTCNN 
# # sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
# # from sort import Sort
# from Learner import face_learner
# from utils import load_facebank, draw_box_name, prepare_facebank

# sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced")
# sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced\retinaface")
# from retinaface.detect_class_hr import FaceDetector
# from face_aligner import FaceAligner

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
#         learner.load_state(conf, r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\work_space\save/models.pth', True, True)
#     else:
#         learner.load_state(conf, r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data/model_ir_se50.pth', True, True)
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

#     video_file_path = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\facebank/abdus-saboor-3-6-right-cam4_PFQ2ptHL.mp4'
#     cap = cv2.VideoCapture(video_file_path)
#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         sys.exit(1)
#     cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print("Video loaded successfully.")
#     print(f"Frames per second (fps): {fps}")
#     video_writer = cv2.VideoWriter(os.path.join(conf.facebank_path, '{}.avi'.format(args.save_name)), cv2.VideoWriter_fourcc(*'XVID'), int(fps), (1280, 720))
#     # sort_tracker = Sort()
#     i = 0
#     frame_count=0
#     frame_start_time1=time.time()

#     while cap.isOpened():
#         isSuccess, frame = cap.read()
#         if isSuccess:
#             start_time=time.time()
#             frame_start_time = time.time()
#             if i < fps * 15:
#                 image = Image.fromarray(frame)
#                 i += 1
#             else:
#                 break

#             try:
#                 dectector_time=time.time()

#                 if conf.detector_name == 'mtcnn':
#                     bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
#                 else:
#                     img = frame
#                     faces, _, bboxes = rface.align_multi(img)
#                 if faces is None:
#                     # print("empty face")
#                     continue
#                 dectector_fps= 1/(time.time()-dectector_time)

#             except:
#                 bboxes = []
#                 faces = []

#             if len(bboxes) == 0:
#                 # print('No face detected in the current frame.')
#                 continue
#             else:
#                 print(i)
#                 # print(f'length of bounding boxes {len(bboxes)}')
#                 if conf.detector_name != 'mtcnn':
#                     algnface = [Image.fromarray(face) for face in faces]
#                     faces = algnface
#                     bboxes = np.array(bboxes)
#                 else:
#                     # bboxes = bboxes[:, :-1]
#                     bboxes = bboxes.astype(int)
#                     # bboxes = bboxes + [-1, -1, 1, 1]

#                 targets_list = [targets]
#                 Recognition_time=time.time()
#                 results, score = learner.infer(conf, faces, targets_list, True)
#                 recognizer_fps= 1/(time.time()-Recognition_time)


#                 # print(np.shape(bboxes))  # Shape of the array
#                 # print(type(bboxes))
#                 # print(bboxes)

                

#                 frame_output_dir = conf.model_name + '_' + conf.detector_name
#                 os.makedirs(frame_output_dir, exist_ok=True)
#                 for idx,bbox in enumerate(bboxes):
#                     bbox = bbox.astype( int)  # Ensure bbox is converted to integer type
#                     if args.score:
#                         frame = draw_box_name(bbox[:4], names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
#                     else:
#                         frame = draw_box_name(bbox[:4], names[results[idx] + 1], frame)
#                     frame_filename = os.path.join(frame_output_dir, f'frame_{i}_{idx}.jpg')
#                     cv2.imwrite(frame_filename, frame)
#                     video_writer.write(frame)
#                     combined_fps= 1/(time.time()-dectector_time)


#                     frame_count+=1
#                     # Overlay Input FPS and Output FPS on the frame
#                     cv2.putText(frame, f"Input FPS: {fps:.2f}", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.putText(frame, f"Output FPS: { combined_fps:.2f}", (800, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
#                     cv2.putText(frame, f"Detector FPS: { dectector_fps :.2f}", (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
#                     cv2.putText(frame, f"Recognizer FPS: { recognizer_fps :.2f}", (800, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
#                     video_writer.write(frame)
#                     frame_count+=1



#         else:
#             break

#         if args.duration != 0:
#             i += 1
#             if i % 25 == 0:
#                 print('{} second'.format(i // 25))
#             if i > 25 * args.duration:
#                 break
#             else:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_POS_FRAMES) + fps))


#     output_fps=frame_count/(time.time()-frame_start_time1)
#     print(f"output fps=",output_fps)

#     cap.release()
#     video_writer.release()
#     print("Video processing complete.")




#*******************************************code*********************

import cv2
from PIL import Image
import argparse
from pathlib import Path
import sys
import os
import numpy as np
import pickle    
import torch
from config import get_config
from mtcnn import MTCNN 
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced")
sys.path.append(r"C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\retinafaced\retinaface")
from retinaface.detect_class_hr import FaceDetector
from face_aligner import FaceAligner

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
        print(9)
        targets, names = load_facebank(conf)
        print('facebank loaded')

    video_file_path = r'C:\Users\123\Desktop\datasets\InsightFace_Pytorch-master (2)\InsightFace_Pytorch-master\data\facebank/abdus-saboor-3-6-right-cam4_PFQ2ptHL.mp4'
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video loaded successfully.")
    print(f"Input video FPS: {input_fps}")
    video_writer = cv2.VideoWriter(os.path.join(conf.facebank_path, '{}.avi'.format(args.save_name)), cv2.VideoWriter_fourcc(*'XVID'), int(input_fps), (1280, 720))
    sort_tracker = Sort() 
    i = 0

    # Set of known tracker IDs
    known_tracker_ids = set()
    tracked_faces = {}

    # To calculate output FPS
    start_time_2 = time.time()
    frame_count = 0

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            start_time=time.time()
            frame_start_time = time.time()
            if i < input_fps * 15:
                image = Image.fromarray(frame)
                i += 1
            else:
                break

            try:
                dectector_time=time.time()
                if conf.detector_name == 'mtcnn':
                    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                else:
                    img = frame
                    faces, _, bboxes = rface.align_multi(img)
                if faces is None:
                    # print("empty face")
                    continue
                dectector_fps= 1/(time.time()-dectector_time)

            except:
                bboxes = []
                faces = []

            if len(bboxes) == 0:
                # print('No face detected in the current frame.')
                continue
            else:
                print(i)
                # print(f'length of bounding boxes {len(bboxes)}')
                if conf.detector_name != 'mtcnn':
                    algnface = [Image.fromarray(face) for face in faces]
                    faces = algnface
                    bboxes = np.array(bboxes)
                else:
                    bboxes = np.array(bboxes)

                    bboxes = bboxes.astype(int)
                trackers = sort_tracker.update(bboxes)

                # List of new faces to process
                new_faces = []
                new_tracker_ids = []

                for track, face in zip(trackers, faces):
                    tracker_id = int(track[4])
                    x1, y1, x2, y2 = map(int, track[:4])
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if tracker_id not in known_tracker_ids or( bbox_width > args.bbox_threshold and bbox_height > args.bbox_threshold):
                        new_faces.append(face)  # Append the face image
                        new_tracker_ids.append(tracker_id)
                        known_tracker_ids.add(tracker_id)

                if new_faces: 
                    recognizer_time =time.time()
                    targets_list = [targets]
                    results, score = learner.infer(conf, new_faces, targets_list, True)
                    recognizer_fps= 1/(time.time()-recognizer_time)

                    
                    for tracker_id, idx in zip(new_tracker_ids, results):
                        identity = "Unknown" if idx == -1 else names[idx.item()+1]
                        tracked_faces[tracker_id] = (identity, score[idx].item() if args.score else None)

                # Annotate and write frame
                for track in trackers:
                    tracker_id = int(track[4])
                    x1, y1, x2, y2 = map(int, track[:4])
                    identity, face_score = tracked_faces.get(tracker_id, ("Unknown", None))
                    bbox = track.astype(int)  # Ensure bbox is converted to integer type
                    if args.score and face_score is not None:
                        frame = draw_box_name(bbox, f"{identity}_{face_score:.2f}", frame)
                    else:
                        frame = draw_box_name(bbox, identity, frame)

                # Calculate output FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                output_fps_combined = 1     / elapsed_time

                # Overlay Input FPS and Output FPS on the frame
                cv2.putText(frame, f"Input FPS: {input_fps:.2f}", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Output FPS: { output_fps_combined :.2f}", (800, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Detector FPS: { dectector_fps :.2f}", (800, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Recognizer FPS: { recognizer_fps :.2f}", (800, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                video_writer.write(frame)
                i += 1

        else:
            break
    
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_POS_FRAMES) + input_fps))
    elapsed_time2 = time.time() - start_time_2
    output_fps = frame_count / elapsed_time2
    print(f"output fps=", output_fps)


    cap.release()
    video_writer.release()
    print("Video processing complete.")

