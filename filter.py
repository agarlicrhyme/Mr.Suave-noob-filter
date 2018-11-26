import numpy as np  
import cv2  
import dlib  
from scipy.spatial import distance as dist  
from scipy.spatial import ConvexHull  
   
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
   
FULL_POINTS = list(range(0, 68))  
FACE_POINTS = list(range(17, 68))  
   
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  
   
detector = dlib.get_frontal_face_detector()  
   
predictor = dlib.shape_predictor(PREDICTOR_PATH)  

def mustache_size(mustache):
   mustacheWidth = dist.euclidean(mustache[0], mustache[3])
   hull = ConvexHull(mustache)
   mustacheCenter = np.mean(mustache[hull.vertices, :], axis = 0)
   mustacheCenter = mustacheCenter.astype(int)

   return int(mustacheWidth), mustacheCenter

def place_mustache(frame, mustacheCenter, mustacheSize):
   mustacheSize = int(mustacheSize * 5)
   x1 = int(mustacheCenter[0,0] - (mustacheSize/4))
   x2 = int(mustacheCenter[0,0] + (mustacheSize/4))  
   y1 = int(mustacheCenter[0,1] - (mustacheSize/2))  
   y2 = int(mustacheCenter[0,1] + (mustacheSize/2)) 
   
   h, w = frame.shape[:2]  
   
   
   if x1 < 0:  
     x1 = 0  
   if y1 < 0:  
     y1 = 0  
   if x2 > w:  
     x2 = w  
   if y2 > h:  
     y2 = h


   mustacheOverlayWidth = x2 - x1  
   mustacheOverlayHeight = y2 - y1  
   
     
   mustacheOverlay = cv2.resize(imgMustache, (mustacheOverlayWidth,mustacheOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask = cv2.resize(orig_mask1, (mustacheOverlayWidth,mustacheOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask_inv = cv2.resize(orig_mask_inv1, (mustacheOverlayWidth,mustacheOverlayHeight), interpolation = cv2.INTER_AREA)  
   
    
   roi = frame[y1:y2, x1:x2]  
   
    
   roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
    
   roi_fg = cv2.bitwise_and(mustacheOverlay,mustacheOverlay,mask = mask)  
   
   roi = frame[y1:y2, x1:x2]  
   
   
   roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
     
   roi_fg = cv2.bitwise_and(mustacheOverlay,mustacheOverlay,mask = mask)  
   
   
   dst = cv2.add(roi_bg,roi_fg)  
   
   
   frame[y1:y2, x1:x2] = dst 
    
   
   
   
def eye_size(eye):  
   eyeWidth = dist.euclidean(eye[0], eye[3])  
   hull = ConvexHull(eye)  
   eyeCenter = np.mean(eye[hull.vertices, :], axis=0)  
   
   eyeCenter = eyeCenter.astype(int)  
   
   return int(eyeWidth), eyeCenter  
   
def place_eye(frame, eyeCenter, eyeSize):  
   eyeSize = int(eyeSize * 1.5)  
   
   x1 = int(eyeCenter[0,0] - (eyeSize/2))  
   x2 = int(eyeCenter[0,0] + (eyeSize/2))  
   y1 = int(eyeCenter[0,1] - (eyeSize/2))  
   y2 = int(eyeCenter[0,1] + (eyeSize/2))  
   
   h, w = frame.shape[:2]  
   
   
   if x1 < 0:  
     x1 = 0  
   if y1 < 0:  
     y1 = 0  
   if x2 > w:  
     x2 = w  
   if y2 > h:  
     y2 = h  
   
    
   eyeOverlayWidth = x2 - x1  
   eyeOverlayHeight = y2 - y1  
   
   
   eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   
   
   roi = frame[y1:y2, x1:x2]  
   
     
   roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
     
   roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)  
   
    
   dst = cv2.add(roi_bg,roi_fg)  
   
   
   frame[y1:y2, x1:x2] = dst  
   

imgEye = cv2.imread('Eye.png',-1)  
imgMustache =cv2.imread('Mustache.png' , -1)  

orig_mask = imgEye[:,:,3]  
orig_mask1 = imgMustache[:,:,3]   

orig_mask_inv = cv2.bitwise_not(orig_mask)  
orig_mask_inv1 = cv2.bitwise_not(orig_mask1)
   
 
imgEye = imgEye[:,:,0:3]  
origEyeHeight, origEyeWidth = imgEye.shape[:2]
imgMustache = imgMustache[:,:,0:3]  
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]  

   

video_capture = cv2.VideoCapture(0)  
   
while True:  
   ret, frame = video_capture.read()  
   
   if ret:  
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
   
     rects = detector(gray, 0)  
   
     for rect in rects:  
       x = rect.left()  
       y = rect.top()  
       x1 = rect.right()  
       y1 = rect.bottom()  
   
       landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  
       mouth = landmarks[MOUTH_OUTLINE_POINTS]
       left_eye = landmarks[LEFT_EYE_POINTS]  
       right_eye = landmarks[RIGHT_EYE_POINTS]  
   

       leftEyeSize, leftEyeCenter = eye_size(left_eye)  
       rightEyeSize, rightEyeCenter = eye_size(right_eye)  
       mustacheSize, mustacheCenter = mustache_size(mouth)
       place_mustache(frame, mustacheCenter, mustacheSize)
       place_eye(frame, leftEyeCenter, leftEyeSize)  
       place_eye(frame, rightEyeCenter, rightEyeSize)  
   
     cv2.imshow("BelikeMR.Suave", frame)  
   
   ch = 0xFF & cv2.waitKey(1)  
   
   if ch == ord('q'):  
     break  
   
cv2.destroyAllWindows()
