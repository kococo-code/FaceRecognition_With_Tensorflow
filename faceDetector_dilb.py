import dlib
import cv2 
import os 

class FaceDetector():
    def __init__(self):
        self.weightPath = os.path.join(os.getcwd(),'src/mmod_human_face_detector.dat')
    def init_model(self):
        face_detector = dlib.cnn_face_detection_model_v1(self.weightPath)
        print("Loaded Face Detector Model")
        return face_detector
    def drawRectangle(self,img,face):
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right()
        h = face.rect.bottom() 
        print(x,y,w,h)
        cv2.rectangle(img, (x,y), (w,h), (0,0,255), 1)
        return img[x:(w),y:(h)]
    def runModel(self,model,img):
        img = cv2.imread(img)
        result = model(img,1)
        roi = []
        for face in result:
            roi.append(self.drawRectangle(img,face))
        if len(result) > 0:
            cv2.imshow('DetectFace',roi[0])
            cv2.waitKey(0)
            cv2.imshow('DetectFace',img)
            cv2.waitKey(0)
if __name__ == "__main__":
    IMG_PATH = './media/image/mask.jpg'
    FaceDetector = FaceDetector()
    model = FaceDetector.init_model()
    FaceDetector.runModel(model,IMG_PATH)
    