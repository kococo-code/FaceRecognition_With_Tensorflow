import cv2 
import glob 
import os 

inputShape = (250,250)

for i in os.listdir('media/raw_face'):
    try:
        os.mkdir('media/face/'+i+'/')
    except:
        pass

    target = glob.glob('media/raw_face/'+i+'/*.jpg')
    print(target)
    idx= 0
    for targetImg in target:
        
        path = 'media/face/'+i+'/'+str(idx)+'.jpg'
        img = cv2.imread(targetImg)
        img = cv2.resize(img,inputShape)
        img = cv2.imwrite(path,img)
        idx +=1