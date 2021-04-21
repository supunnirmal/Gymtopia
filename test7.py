# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import matplotlib.pyplot as plt 
import numpy as np
from scipy.spatial import procrustes
from flask import Flask, request, jsonify
import json
import pandas as pd
from flask_ngrok import run_with_ngrok

import urllib.request
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore , storage
import threading
import datetime
    
# Use the application default credentials
#cred = credentials.ApplicationDefault()

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
            # Windows Import
    if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

        # Flags
parser = argparse.ArgumentParser()
        #parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, #png, bmp, etc.).")
parser.add_argument("--image_path", default="", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    
args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
    
cred = credentials.Certificate("serviceaccount.json")

firebase_admin.initialize_app(cred, {
  'projectId': 'gymtopia-app',
  'storageBucket': 'gymtopia-app.appspot.com'
})

db = firestore.client()

#firebase = pyrebase.initialize_app(config)
#storage = firebase.storage()
#storage.child("images/example.jpg")
#app = Flask(__name__)
#run_with_ngrok(app)



def find_transformation(model_features, input_features):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))]) 
    unpad = lambda x: x[:, :-1]

    input_counter = 0

    nan_indices = []

    #print("inputttt: " , input_features)

    input_features_zonder_nan = []
    model_features_zonder_nan = []
    for in_feature in input_features:
        if (in_feature[0] == 0) and (in_feature[1] == 0): # is a (0,0) feature
            nan_indices.append(input_counter)
        else:
            input_features_zonder_nan.append([in_feature[0], in_feature[1]])
            model_features_zonder_nan.append([model_features[input_counter][0], model_features[input_counter][1]])
        input_counter = input_counter+1
    
    input_features = np.array(input_features_zonder_nan)
    model_features = np.array(model_features_zonder_nan)

    # padden:
    # naar vorm [ x x 0 1]
    Y = pad(model_features)
    X = pad(input_features)
    
    A, res, rank, s = np.linalg.lstsq(X, Y)
    transform = lambda X: unpad(np.dot(pad(X), A))
    input_transform = transform(input_features)

    A[np.abs(A) < 1e-10] = 0  # set really small values to zero

    return (input_transform)
    
def split_in_face_legs_torso(features):
    # torso = features[2:8]   #zonder nek
    torso = features[1:8]   #met nek  => if nek incl => compare_incl_schouders aanpassen!!
    legs = np.vstack([features[8:15], features[19:25]])
    face = np.vstack([features[0], features[15:19]])

    return (face, torso, legs)
    
def timeset(millis):
    #millis=input("Enter time in milliseconds ")
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24

    return ("%d:%d:%d" % (hours, minutes, seconds))

def unsplit(face, torso, legs):
    whole = np.vstack([face[0], torso, legs[0:8], face[1:5],legs[8:14]])

    return whole
    
def normalize(features):
    #print(features)
    distance = np.linalg.norm(features[1] - features[8])
    #print('distance - ', distance)
    return distance
    
def keyposition_img(min1 , min2 , video,mistake,queue):


    comment = {
        1:'lift your upper arm until parallel to the floor , then flare your elbows until the forearm and upper arm is at 90 degree,\nArms form a square.',
        2:'lift your upper arm until parallel to the floor, then narrow gap between your elbows until the forearm and upper arm is at 90 degree,\nArms form a square.',
        3:'Lower your upper arm until parallel to the floor, then flare your elbows until the forearm and upper arm is at 90 degree,\nArms form a square.',
        4:'Narrow the gap between your wrists. Lower your upper arm until parallel to the floor .The forearm and upper arm is at 90 degree,\nArms form a square.',
        5:'lift your upper arm until it\'s parallel to the floor. It will form 90 degree angle with the forearm,\nArms form a square.',
        6:'Lower your upper arm until parallel to the floor, then it will make a 90 degree angle with the forearm,\nArms form a square.',
        7:'Flare the gap between the wrist until the forearm and upper arm is at 90 degree,\nArms form a square',
        8:'Narrow  the gap between your wrists until  the forearm and upper arm is at 90 degree,\nArms form a square',
        9:'Don’t lock your elbows and don’t flare the forearm,\nNarrow the gap between your wrists.', 
        10:'Straighten both your forearm and the upper arm.', 
        11:'Straighten the upper arm.',
        12:'Straighten your forearm.Avoid clanking the weights. flare the gap between your wrists'}


    cap = cv2.VideoCapture(video)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        
    i=0
    j=0
    k=0    
    #output = []
    out = []
    #start = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            #frame = cv2.resize(frame,(426,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            bucket = storage.bucket()
            
            if(min1[i][1] == t):
                score = int(mistake[k][0])
                #cv2.imshow('Key1'+str(score),frame)
                #name = str(i) + 'min1.jpg'
                if(score!=0):
                    
                    # Starting OpenPose
                    #opWrapper = op.WrapperPython()
                    #opWrapper.configure(params)
                    #opWrapper.start()
                    #name = queue['user']+'-'+str(round(t,0))
                    #cv2.imwrite(name+'.jpg', frame)
                    

    # Process Image
                    datum = op.Datum()
                    #imageToProcess = cv2.imread(name+'.jpg')
                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
                    #print("Body keypoints: \n" + str(datum.poseKeypoints))
                    #im = cv2.resize(datum.cvOutputData, (1096, 779)) 
                    name = queue['user']+'-'+str(round(t,0))
                    cv2.imwrite(name+'.jpg', datum.cvOutputData)
                    ts = time.time()
                
                    imagePath = name+".jpg"
                
                    #bucket = storage.bucket()
                    imageBlob = bucket.blob('feedback'+'/'+queue['user']+'/'+queue['exercise_id']+'/'+str(queue['timestamp'])+'/'+name+".jpg")
                    imageBlob.upload_from_filename(imagePath)
                    url = imageBlob.public_url
                    #temp = datetime.datetime.fromtimestamp(t / 1000).strftime('%H:%M:%S')
    
                    feedback = {'error':comment[score], 'timestamp':timeset(t),'url': url}
                    out.append(feedback)
                    print(out)
                
                if(i+1<len(min1)):
                    i=i+1
                if(k+1<len(mistake)):
                    k=k+1    
                
            if(min2[j][1] == t) :
               
                score = int(mistake[k][0])
                #name = str(j) + 'min2.jpg'
                #cv2.imshow('Key2'+str(score),frame)
                #cv2.imwrite(name, frame)
                if(score!=0):
                
                    #opWrapper = op.WrapperPython()
                    #opWrapper.configure(params)
                    #opWrapper.start()
                    #name = queue['user']+'-'+str(round(t,0))
                    #cv2.imwrite(name+'.jpg', frame)
                    
                    print(1)
    # Process Image
                    datum = op.Datum()
                    #imageToProcess = cv2.imread(name+'.jpg')
                    datum.cvInputData = frame 
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    print(2)
                    name = queue['user']+'-'+str(round(t,0))
                    cv2.imwrite(name+'.jpg', datum.cvOutputData)
                    ts = time.time()
                    print(3)
                    imagePath = name+".jpg"
                    imageBlob = bucket.blob('feedback'+'/'+queue['user']+'/'+queue['exercise_id']+'/'+str(queue['timestamp'])+'/'+name+".jpg")
                    imageBlob.upload_from_filename(imagePath)
                    url = imageBlob.public_url
                    
                    #temp = datetime.datetime.fromtimestamp(int(t) / 1000).strftime('%H:%M:%S')
                    #print(temp)
                    feedback = {'error':comment[score], 'timestamp':timeset(t), 'url': url}
                    out.append(feedback)
                    print(out)
                
                if(j+1<len(min2)):
                    j=j+1
                
                if(k+1<len(mistake)):
                    k=k+1    
                    
            cv2.waitKey(0)
        else: 
            break
    
    #result = (min1[:,1],min2[:,1],mistake)
    #output.append(result)
    #print(output)
    return out
        
def keypose_dist(distance1,distance2,video,queue):
    print('11')
    x = 0
    y = 0
    z = 0
    mincount = 0
    wrong = 0
    output = []
                
    min1 = [[1,1,[[0]*7]*2]]*100
    min2 = [[1,1,[[0]*7]*2]]*100
    min1low=[[1,1,[[0]*7]*2]]
    mistake = [0]*100
    
    for i in range((len(distance1)-1)) : 
   
        if( (distance2[i][0]>distance2[i+1][0]) & (distance1[i][0] > distance2[i][0] ))  :
        
            if((x != y) & ((distance2[i+1][2][1][3] < distance2[i+1][2][1][1])&(distance2[i+1][2][1][6] < distance2[i+1][2][1][4]))):

                if ((distance2[i+1][2][1][3] !=0) & (distance2[i+1][2][1][1]!=0)&(distance2[i+1][2][1][6]!=0) &( distance2[i+1][2][1][4]!=0)):

                    if(min2[x][0]!=1):

                        if((min2[x][2][1][3] >= (distance2[i+1][2][1][3]))):#or(min2[x][2][1][6] > (distance2[i+1][2][1][6]))):
                            min2[x] = (distance2[i+1][0],distance2[i+1][1],distance2[i+1][2])

                    else:
                        min2[x] = (distance2[i+1][0],distance2[i+1][1],distance2[i+1][2])

            if((min1[y][0]!=1) & (y == x)):

                if(mincount == 1):
                    min1[y] = min1low
                    

                if((min1[y][2][1][2]>(min1[y][2][1][1]+0.1))& (min1[y][2][0][3]>(min1[y][2][0][2]+0.1))) : # key 1 below the limit and moving closure
                    mistake[z] = 1
                    
                elif((min1[y][2][1][2]>(min1[y][2][1][1]+0.1))& (min1[y][2][0][3]<(min1[y][2][0][2]-0.1))) : # key 1 below the limit and moving away
                    mistake[z] = 2
                    
                elif((min1[y][2][1][2]<(min1[y][2][1][1]-0.1)) &(min1[y][2][0][3]>(min1[y][2][0][2]+0.1))) : # key 1 above the limit and moving closure
                    mistake[z] = 3
                    
                elif((min1[y][2][1][2]<(min1[y][2][1][1]-0.1)) & (min1[y][2][0][3]<(min1[y][2][0][2]-0.1))) : # key 1 above the limit and moving away
                    mistake[z] = 4
                    
                elif(min1[y][2][1][2]>(min1[y][2][1][1]+0.1)): # key 1 shoulders below the limit
                    mistake[z] = 5

                elif(min1[y][2][1][2]<(min1[y][2][1][1]-0.1)): #key 1 shoulders above the limit
                    mistake[z] = 6
                    
                elif(min1[y][2][0][3]>(min1[y][2][0][2]+0.1)): # key 1 palm moving closer
                    mistake[z] = 7
                    
                elif(min1[y][2][0][3]<(min1[y][2][0][2]-0.1)) : # key 1 palm moving away
                    mistake[z] = 8
                    
                else : 
                    mistake[z] = 0
                    
                z=z+1
                y=y+1
                mincount = 0
        
        if((distance1[i][0]>distance1[i+1][0]) & (distance1[i][0] < distance2[i][0]) & (min1[y][0] > distance1[i+1][0]) ) :
            
            if ((x == y) & (mincount == 0)):
                min1[y] = (distance1[i+1][0],distance1[i+1][1],distance1[i+1][2])
                    
            if((min2[x][0]!=1)  & (x != y)):
                
                if((min2[x][2][0][2]< min2[x][2][0][1]-0.1)&(min2[x][2][0][3]< min2[x][2][0][2]-0.1)): # key 2 elbow moving away and palm moving away
                    mistake[z] = 9
                    
                elif((min2[x][2][0][2]< min2[x][2][0][1]-0.1)& (min2[x][2][0][3]> min2[x][2][0][2]+0.1)): # key 2 elbow moving away and palm moving closure
                    mistake[z] = 10
                    
                #if(min2[x][2][0][2]> min2[x][2][0][1]+0.1): # key 2 elbow moving closure 
                #    z=z+1
                    
                elif(min2[x][2][0][2]< min2[x][2][0][1]-0.1): # key 2 elbow moving away
                    mistake[z] = 11
                    
                #if(min2[x][2][0][3]< min2[x][2][0][2]-0.1): # key 2 palm moving away
                #    z=z+1
                    
                elif(min2[x][2][0][3]> min2[x][2][0][2]+0.1): # key 2 palm moving closure
                    mistake[z] = 12
                    
                else :
                    mistake[z] = 0
                    
                z=z+1
                x=x+1
        
        if((distance1[i][0]<distance1[i+1][0]) & (distance1[i][0] < distance2[i][0]) & ((min1[y][2][1][2]<distance1[i][2][1][2]))) :
        
            if ((x == y) & (x!=0) & (min1[y][2][1][2] !=0)&(distance1[i][2][1][2]!=0) ):
                if((mincount != 0) ):
                    if((min1low[2][1][2]<distance1[i][2][1][2]) & (min1low[2][1][2]!=0)):
                        min1low = (distance1[i][0],distance1[i][1],distance1[i][2])
                    
                if(mincount == 0):
                    mincount = 1
                    min1low = (distance1[i][0],distance1[i][1],distance1[i][2])
                    
            if( distance1[i][0] > 0.45):
                mincount = 0
        
    #    if(y>0 & (distance1[i][2][0][2]!=0) & (distance1[i][2][0][3]!=0) & (distance1[i][2][0][5]!=0) &( distance1[i][2][0][6]!=0)):
    #        print('1')
    #        if(((distance1[i][2][0][2]+50 < distance1[i][2][0][3]) or (distance1[i][2][0][5]-50 > distance1[i][2][0][6]))and wrong ==0 ):
    #            mistake[z] = distance1[i][1],1,y,x
    #            wrong = 1
    #        print('2')
    #        if(((distance1[i][2][0][2]-20 > distance1[i][2][0][3]) or (distance1[i][2][0][5]+20 < distance1[i][2][0][6])) and wrong ==0):
    #            mistake[z] = distance1[i][1],0,y,x 
    #            wrong = 1
            
    #        print('3')
    #        if((distance1[i][2][0][2]+50 >= distance1[i][2][0][3]) and (distance1[i][2][0][5]-50 <= distance1[i][2][0][6])and((distance1[i][2][0][2]-20 <= distance1[i][2][0][3]) and (distance1[i][2][0][5]+20 >= distance1[i][2][0][6]))):
                
    #            if (wrong == 1):
    #                mistake[z] = round((mistake[z][0]+distance1[i][1])/2,-2), mistake[z][1],mistake[z][2],mistake[z][3]
    #                z=z+1
    #                wrong =0
    #        print('4')
    min1 = np.array(min1)
    min2 = np.array(min2)
    mistake = np.array(mistake)
    
    print(x)
    print(y)
    #print(mistake)
    if(x == y):
        y=y+1
    min1.resize(y,3,refcheck=False)
    min2.resize(x,3,refcheck=False) #before end of the video person comes to start position 
    
    mistake.resize(z,1,refcheck=False)
    print(len(mistake))
    print(mistake)
    #print(mistake[0][1])
    #print('--------------------------------------------')
    #print(min2)
    
    print("rep count = ", y-1)
    
    out = keyposition_img(min1,min2,video,mistake,queue)
    #plt.scatter(min1[:,1], min1[:,0], color ='black')
    #plt.scatter(min2[:,1], min2[:,0], color ='yellow')
    print(out)
    #var = min1[y-1][1] - min1[0][1]
    #temp = datetime.datetime.fromtimestamp(var / 1000).strftime('%H:%M:%S')
    #print (temp)
    result = {'reps':y-1,'calories': 0 , 'minutes': timeset(min1[y-1][1] - min1[0][1]), 'errors' : out}
    
    #result = (min1[:,1],min2[:,1],mistake)
    #output.append(result)
    #return (min1[:,1],min2[:,1],mistake)
    print(result)
    return result
    
    
x = 0
extract = []
def on_snapshot(col_snapshot, changes, read_time):
    
    global x
    global queue
   
    print(u'Callback received query snapshot.')
    print(u'Current cities in California: ')
    for change in changes:
        if change.type.name == 'ADDED':
            #print(f'New city: {change.document.id}')
            print(change.document.to_dict())
            item = change.document.to_dict()
            item['id'] = change.document.id
            extract.append(item)
            x = 1
        #elif change.type.name == 'MODIFIED':
        #    print(f'Modified city: {change.document.id}')
        #elif change.type.name == 'REMOVED':
        #    print(f'Removed city: {change.document.id}')
            #delete_done.set()
#@app.route('/vid/<path:path>',methods = ['GET'])    
def model():
    global x
    global extract
    
    queue = extract
    extract.clear()
    delete_done = threading.Event()
    #urllib.request.urlretrieve(url_link, 'video_name.mp4') 
    col_query = db.collection(u'processing_queue')

    query_watch = col_query.on_snapshot(on_snapshot)


#print('xx')
#query_watch.unsubscribe()
    print(x)
    while x==0:
        #time.sleep(1)
        delete_done.wait(timeout=10)
        print('processing...')
    
    print(x)
    print(len(queue))
    print(queue)
    print(queue[0]['url'])

    x =0
    for l in range(len(queue)):
        urllib.request.urlretrieve(queue[l]['url'], queue[l]['user']+'-'+str(queue[l]['timestamp'])+'.mp4') 
        input = queue[l]['user']+'-'+str(queue[l]['timestamp'])+'.mp4'
        
        #db.collection(u'processing_queue').document(queue[i]['id']).delete()
        try:
        
            #storage.child(path).download(path)

            # Import Openpose (Windows/Ubuntu/OSX)
            
        # Add others in path?
        #for i in range(0, len(args[1])):
        #    curr_item = args[1][i]
        #    if i != len(args[1])-1: next_item = args[1][i+1]
        #    else: next_item = "1"
        #    if "--" in curr_item and "--" in next_item:
        #        key = curr_item.replace('-','')
        #        if key not in params:  params[key] = "1"
        #    elif "--" in curr_item and "--" not in next_item:
        #        key = curr_item.replace('-','')
        #        if key not in params: params[key] = next_item

    
        # Starting OpenPose
           # opWrapper = op.WrapperPython()
           # opWrapper.configure(params)
           # opWrapper.start()
    

        # Process Image
            datum = op.Datum()
            imageToProcess = cv2.imread("20.png")
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
            height = imageToProcess.shape[0]
            width = imageToProcess.shape[1]
    
            height = imageToProcess.shape[0]
            width = imageToProcess.shape[1]
    
            d = normalize(datum.poseKeypoints[0,:,0:2])
        
            img2=[[0]*2]*25
            for i in range(25):
            # img2[i]=(((((datum.poseKeypoints)[0][i][0])/width)*1000)/d,((((datum.poseKeypoints)[0][i][1])/height)*1000)/d)
                img2[i]=(((datum.poseKeypoints)[0][i][0])/d,((datum.poseKeypoints)[0][i][1])/d)
            
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #im = cv2.resize(datum.cvOutputData, (480, 480))  
        #cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
            cv2.waitKey(0)
    
            datum = op.Datum()
            imageToProcess = cv2.imread("21.png")
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", im)
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
    
            height = imageToProcess.shape[0]
            width = imageToProcess.shape[1]
    
            d = normalize(datum.poseKeypoints[0,:,0:2])
    
            img1=[[0]*2]*25
            for i in range(25):
            #img1[i]=(((((datum.poseKeypoints)[0][i][0])/width)*1000)/d,((((datum.poseKeypoints)[0][i][1])/height)*1000)/d)
                img1[i]=(((datum.poseKeypoints)[0][i][0])/d,((datum.poseKeypoints)[0][i][1])/d)
    
            distance1 = []
            distance2 = []
    
            video = input

            cap = cv2.VideoCapture(video)
            if (cap.isOpened()== False): 
                print("Error opening video stream or file")
        
            k=0
            while(cap.isOpened()):
            
                ret, frame = cap.read()
                if ret == True:
                #frame = cv2.resize(frame,(426,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    datum = op.Datum()
                #cv2.imshow('xx',frame)
            
                    datum.cvInputData = frame
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    if datum.poseKeypoints is not None:
                        if(datum.poseKeypoints.size == 0):
                            continue
                
                        d = normalize(datum.poseKeypoints[0,:,0:2])
                
                        height = frame.shape[0]
                        width = frame.shape[1]
                        j = 0
                        if d == 0 :
                            d = 1
                        c = [[0]*2]*7
                        model1 = [[0]*2]*7
                        model2 = [[0]*2]*7
                        for i in range(7):
                            if datum.poseKeypoints[0][i+1][2] > 0 :
                                j = j+1
                            #c[j-1] = (((((datum.poseKeypoints)[0][i+1][0]/width)*1000)/d),((((datum.poseKeypoints)[0][i+1][1])/height)*1000)/d)
                                c[j-1] = (((datum.poseKeypoints)[0][i+1][0]/d),((datum.poseKeypoints)[0][i+1][1])/d)
                        
                                model2[j-1] = (img2[i+1][0], img2[i+1][1])
                                model1[j-1] = (img1[i+1][0], img1[i+1][1])
                
                        input = np.array(c)
                    #print(datum.poseKeypoints)
                    # print('--------------------------------------------')
                    #cv2.imshow('xx',frame)
                
                    #origin = (((((datum.poseKeypoints)[0][1:8,0]/width)*1000)/d),((((datum.poseKeypoints)[0][1:8,1]/height)*1000)/d))
                        origin = (((datum.poseKeypoints)[0][1:8,0]/d),((datum.poseKeypoints)[0][1:8,1])/d)
                
                    #plt.scatter(datum.poseKeypoints[0][1:8,0], datum.poseKeypoints[0][1:8,1], color ='blue')
                    #plt.show()
                    # print(origin)
                        input.resize(j,2,refcheck=False)
                        if(input.size == 0):
                            continue
                
                        keypose2 = np.array(model2)
                        keypose2.resize(j,2,refcheck=False)
                        result_torso = find_transformation(keypose2,input)
                        dist2 = np.linalg.norm(keypose2-result_torso)
                
                        print(dist2)
                        keypose1 = np.array(model1)
                        keypose1.resize(j,2,refcheck=False)
                        result_torso = find_transformation(keypose1,input)
                        dist1 = np.linalg.norm(keypose1-result_torso)
            
                        time = cap.get(cv2.CAP_PROP_POS_MSEC)
                        plt.scatter(time, dist1, color ='red')
                        plt.scatter(time, dist2, color ='blue')
                
                    #plt.scatter(time, origin[1][3], color ='green')
    
                        pair1 = (dist1,time,origin)
                        pair2 = (dist2,time,origin)
                        distance1.append(pair1)
                        distance2.append(pair2)
                
                        cv2.waitKey(0)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                else: 
                    break
        #cap.release()
    
        #key1,key2,mistake = keypose_dist(distance1,distance2,video)
            
        #plt.show()
            cap.release()
            cv2.destroyAllWindows()
            print('xx')
            if((len(distance1)>0)&(len(distance2)>0)):
                print('yy')
                print(queue[l])
                result = keypose_dist(distance1,distance2,video,queue[l])
                print('11')
                output = db.collection(u'users').document(queue[l]['user'])
                #print(44)
                #queue[l].pop('id')
                #queue[l].pop('user')
                #queue[l].pop('type')
                #print(33)
                #print(queue)
                remove = []
                remove.append({u'exercise_id':queue[l]['exercise_id'],u'exercise_title':queue[l]['exercise_title'],u'timestamp':queue[l]['timestamp'],u'url':queue[l]['url']})
                output.update({u'classic_exercises.sessions': firestore.ArrayRemove(remove)})
                
                print('22')
                upload = []
                upload.append({u'exercise_id': queue[l]['exercise_id'],
                    u'exercise_title':queue[l]['exercise_title'],
                    u'timestamp': queue[l]['timestamp'],
                    u'url': queue[l]['url'],
                    u'feedback': result})
                
                output.update({u'classic_exercises.sessions': firestore.ArrayUnion(upload)})

                
            else:
                print('no any keypoints detected')
                
                
                
        #print(key1[0])
        #print(key2)
        #print(mistake)
        
        #output = {"key1" : key1[0], "key2" : key2[0], "mistake" : mistake[0] }
      
        #    print(output)
        #    result = pd.Series(output).to_json(orient='values')
        #lists = output.tolist()
        #json_str = json.dumps(output)
        #print(json_str)
            #return jsonify({'output':result})
    
    
        except Exception as e:
            print(e)
            sys.exit(-1)
    
    queue.clear()
    
    model()

#if __name__ == '__main__':
#    app.run()

if __name__ == "__main__":
    model()