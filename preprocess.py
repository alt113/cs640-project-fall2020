import csv
import cv2
import numpy as np
import os

LABELS = ['positive', 'neutral', 'negative']
label_dict = {}
with open('local_preprocessing/labels.csv', 'r', newline='') as labels:
    label_reader = csv.reader(labels, delimiter=' ', quotechar='|')
    next(label_reader)
    for row in label_reader:
        entry = row[0].split(',')
        label_dict[entry[0]] = entry[1].lower()
        
print(label_dict)

curr_dir = os.getcwd()
vid_clips_2020 = curr_dir + '/local_preprocessing/example_clips/'

def preprocess_frame(frame, crop_pad=5):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces using open CV's
    faces_detected = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    # if not 1 face, disregard this frame
    if len(faces_detected) != 1:
        return False
    
    (x, y, w, h) = faces_detected[0]
    # crop the frame around the face
    frame_cropped = frame[y-crop_pad+1:y+h+crop_pad, x-crop_pad+1:x+w+crop_pad]
    return frame_cropped


def preprocess_clip(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video = directory + filename
            vidcap = cv2.VideoCapture(video)
            success,image = vidcap.read()
            count = 0
            new_dir = '{}/data/{}/'.format(curr_dir, label_dict[filename])
            while success:
                frame_file = '{}{}_frame{}_{}.jpg'.format(new_dir, os.path.splitext(filename)[0], count, label_dict[filename])
                if not os.path.exists(os.path.dirname(frame_file)):
                    try:
                        os.makedirs(os.path.dirname(frame_file))
                    except OSError as exc: # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                
                new_frame = preprocess_frame(image)
                if (type(new_frame) == np.ndarray):
                    cv2.imwrite(frame_file, new_frame)     # save frame as JPEG file
                else:
                    break # once you stop detecting one face, don't take more frames
                success,image = vidcap.read()
                count += 1
            print('Split video', video)
            
preprocess_clip(vid_clips_2020)