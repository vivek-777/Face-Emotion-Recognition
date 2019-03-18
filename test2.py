import cv2
import glob

import cntk
from cntk.layers import Convolution2D, MaxPooling ,Activation
from tflearn.layers.normalization import local_response_normalization

#######################################################################################################


"""FACE DETECTION PART"""


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
emotions = ['angry','disgust','sad','fear','surprise','happy','neutral']
features = []

def detect_faces(emotion):
    files = glob.glob("jaffe/%s/*" %(emotion))
    filenumber = 0
    for f in files:
        gray = cv2.imread(f)
        """gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) #Convert image to grayscale"""
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        """if len(face) == 1:
            facefeatures = face
        else:
            facefeatures = ""  """
        for (x, y, w, h) in face:
            gray = gray[y:y+h, x:x+w]
            try:
                out = cv2.resize(gray, (224, 224))
                features.append(Alexnet(out))
                cv2.imwrite("dataset/%s/%s.jpg" %(emotion,filenumber), out)
            except:
               pass
        filenumber += 1
    print(features)

for emotion in emotions:
    detect_faces(emotion)
    
#######################################################################################################


def Alexnet(network):
    network = Convolutional2D(network, (11,11), 96, init=normal(0.01), pad=False, strides=4, bias=True, name="conv1")
    network = Activation(network, activation=relu, name='relu1')
    network = MaxPooling(network, (3,3), strides=2, name='pool1')
    network = local_response_normalization(network)
    
    network = Convolutional2D(network, (5,5), 256, init=normal(0.01), pad=False, strides=1, bias=True, name="conv2")
    network = Activation(network, activation=relu, name='relu2')
    network = MaxPooling(network, (3,3), strides=2, pad=False, name='pool2')
    network = local_response_normalization(network)
    
    network = Convolutional2D(network, (3,3), 384, init=normal(0.01), pad=False, strides=1, bias=True, name="conv3")
    network = Activation(network, activation=relu, name='relu3')

    network = Convolutional2D(network, (3,3), 384, init=normal(0.01), pad=False, strides=1, bias=True, name="conv4")
    network = Activation(network, activation=relu, name='relu4')

    network = Convolutional2D(network, (3,3), 256, init=normal(0.01), pad=False, strides=1, bias=True, name="conv5")
    network = Activation(network, activation=relu, name='relu5')
    network = MaxPooling(network, (3,3), strides=2, pad=False, name='pool5')
    return(network)

#######################################################################################################

