import cv2
import os
import mediapipe as mp
import argparse

# create dirs
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def process_img(img, face_detection):
    H,W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None: # atleast 1 face is found
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            
            # blur face
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (25,25))
    return img


args = argparse.ArgumentParser()

# for IMAGES
#args.add_argument("--mode", default='image')
#args.add_argument("--filePath", default='./data/test1.png')

# for VIDEO
args.add_argument("--mode", default='video')
args.add_argument("--filePath", default='./data/vid.mp4')

# for WEBCAM
#args.add_argument("--mode", default='webcam')


args = args.parse_args()

# detect face
mp_face = mp.solutions.face_detection

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5,) as face_detection:
    
    if args.mode in ["image"]:
        # read img
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detection)
        # save img
        cv2.imwrite(os.path.join('.','output','test1_out.png'), img)

    elif args.mode in ["video"]:
        # read Video
        cap = cv2.VideoCapture(args.filePath)
        ret , frame = cap.read()
        output_vid = cv2.VideoWriter(os.path.join(output_dir,'vid_output.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     25,
                                     (frame.shape[1], frame.shape[0]))
        while ret:
            frame = process_img(frame, face_detection)
            # save Video
            output_vid.write(frame)
            ret, frame = cap.read()
        cap.release()
        output_vid.release()
    
    elif args.mode in ["webcam"]:
        # read Webcam
        cap = cv2.VideoCapture(0)
        ret , frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('frame',frame)
            if cv2.waitKey(25) & 0xFF==ord('q'):
                cv2.destroyAllWindows()
                break
            ret, frame = cap.read()
        cap.release()