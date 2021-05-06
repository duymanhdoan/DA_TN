import cv2
from models_ai import FaceRecognize
import args 
#---- init -- models face recognition. 



class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.checker = FaceRecognize(args)
        self.targets , self.names = self.checker.update_facedb(args)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        if success:
            image = self.checker.face_recognize(image, args, self.targets, self.names)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
            # print(type(image))
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
       
        