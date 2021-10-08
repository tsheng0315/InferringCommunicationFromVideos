import sys
import cv2
import face_recognition
import scipy.optimize as opt
import json
import csv
import math
import dlib
import pandas as pd
from pathlib import Path
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt
from csv import reader


from person import Person

video_id = sys.argv[1]

# video = f'./videosLabelled/1/webcam.mp4'
# sourceFolder = f'./run-result-hands/1/'
# studentFolder = f'./student-faces/1/'



video = f'./videosLabelled/{video_id}/webcam.mp4'
# openpose result 
sourceFolder = f'./run-result-hands/{video_id}/'
# ./student-faces/1/A_0.jpg, A_1.jpg
studentFolder = f'./student-faces/{video_id}/'

resultFolder = f'./run-result-4/{video_id}/'
resultJson = f'./data/{video_id}.json'
resultCsv = f'./data/{video_id}.csv'

data = genfromtxt(f'./videosLabelled/{video_id}/data.csv', delimiter=',', names=True)
with open(f'./videosLabelled/{video_id}/gaze_data.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    list_of_tuples = list(map(tuple, csv_reader))
    gaze_data=list_of_tuples

pd.read_csv(f'./videosLabelled/{video_id}/gaze_data.csv')  



jsonFile = open(resultJson,'w')
csvFile = open(resultCsv, 'w')
csvWritter  = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

print(studentFolder)

data_result_folder = f'./run-result-id-data/{video_id}/'

def getEncoding(img):
    # ./student-faces/1/A_0.jpg, A_1.jpg
    loaded_image = face_recognition.load_image_file(studentFolder + img)
    return face_recognition.face_encodings(loaded_image)[0]

names = [
    "A",
    "B"
]

counts = { "A": 0, "B": 0 }

def loadEncodings():
    known_faces = []
    ids = []
    # ./student-faces/1/A_0.jpg, A_1.jpg
    images = [f for f in listdir(studentFolder) if isfile(join(studentFolder, f))]
    
    # loop through images in folder ./student-faces/1/A_0.jpg, A_1.jpg
    for image in images:
        
        for id, name in enumerate(names):
            print(id,name)
            # (0,A) (1,B)
            if image.startswith(name):
                # A_0.jpg
                known_faces.append(getEncoding(image))
                # known_faces append with encoding of A_0.jpg
                ids.append(id)
                # ids=[0,0,0,1,1]

    return known_faces, ids
        
known_faces, ids = loadEncodings()

persons = []
# UNKNOWN = 2
# last_known_positions = [
#     None,
#     None
# ]

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)

frames = 0
snapshots = 0
# snapshotEachSeconds = 1.0
currentDataRow = 0

while(cap.isOpened()):
    
    # while frames < 0:
    #     cap.read()
    #     frames += 1
    #     snapshots += 1

    ret, frame = cap.read()
    videoTime = frames / fps
    if ret==False:
        break

    # load persons
    # ./run-result-hands/1/webcam_000frameNumber_keypoints.json
    # with open(sourceFolder + "webcam_" + "000{:09d}".format(frames) + "_keypoints.json") as file:
    with open(sourceFolder + "webcam_" + "{:04d}".format(frames) + "_keypoints.json") as file:
        openPose = json.load(file)
        # print(type(openPose))

    
    def filter(arr):
        return arr[:27] + arr[45:57]

    for person in openPose["people"]:
        del person["face_keypoints_2d"]
        del person["face_keypoints_3d"]
        del person["pose_keypoints_3d"]
        del person["hand_left_keypoints_3d"]
        del person["hand_right_keypoints_3d"]
        person["pose_keypoints_2d"] = filter(person["pose_keypoints_2d"])


    # from openpose select upper body part
#     {
#     "-1":{
#        "pose_keypoints_2d":[
#           [
#              1186.51,
#              200.629,
#              0.881209,
#              1330.9,
#              362.455,
#              0.776491,
#              1213.24,
#              344.81,
#              0.725208,
#              1230.78,
#              542.034,
#              0.474453,
#              1163.11,
#              597.986,
#              0.578756,
#              1448.52,
#              371.413,
#              0.604975,
#              1501.57,
#              627.498,
#              0.664912,
#              1378.05,
#              818.755,
#              0.597444,
#              1289.66,
#              786.338,
#              0.301716,
#              1186.68,
#              168.287,
#              0.0750078,
#              1210.1,
#              171.17,
#              0.884849,
#              0,
#              0,
#              0,
#              1307.32,
#              183.025,
#              0.862849
#           ]
#        ],
#        "hand_right_keypoints_2d":[
#           [
#              1166.93,
#              596.112,
#              0.0222111,
#              1172.74,
#              567.058,
#              0.0274886,
#              1181.45,
#              561.247,
#              0.0338103,
#              1140.05,
#              595.386,
#              0.0604027,
#              1125.52,
#              619.356,
#              0.0733598,
#              1150.22,
#              588.122,
#              0.0406811,
#              1133.51,
#              617.903,
#              0.0363625,
#              1135.69,
#              616.451,
#              0.048873,
#              1143.68,
#              699.983,
#              0.0500558,
#              1169.1,
#              598.291,
#              0.0181911,
#              1147.31,
#              616.451,
#              0.0320927,
#              1142.23,
#              618.63,
#              0.034075,
#              1148.04,
#              618.63,
#              0.0230407,
#              1169.83,
#              600.47,
#              0.0195863,
#              1087.75,
#              705.794,
#              0.0271704,
#              1155.3,
#              634.61,
#              0.023817,
#              1148.77,
#              619.356,
#              0.0192149,
#              1084.12,
#              702.162,
#              0.0227933,
#              1080.49,
#              710.878,
#              0.0530098,
#              1079.04,
#              708.699,
#              0.0373135,
#              1077.58,
#              702.888,
#              0.0156758
#           ]
#        ],
#        "hand_left_keypoints_2d":[
#           [
#              1367.03,
#              810.875,
#              0.304759,
#              1334.42,
#              835.819,
#              0.355733,
#              1327.7,
#              877.073,
#              0.291528,
#              1333.46,
#              907.773,
#              0.255168,
#              1340.17,
#              920.245,
#              0.209693,
#              1337.29,
#              884.748,
#              0.561979,
#              1341.13,
#              912.57,
#              0.521573,
#              1344.97,
#              920.245,
#              0.337767,
#              1348.81,
#              923.123,
#              0.281959,
#              1358.4,
#              886.667,
#              0.592188,
#              1364.16,
#              908.732,
#              0.460129,
#              1370.87,
#              920.245,
#              0.513306,
#              1378.55,
#              921.204,
#              0.399338,
#              1379.51,
#              883.788,
#              0.620748,
#              1380.47,
#              902.017,
#              0.643891,
#              1388.14,
#              915.448,
#              0.785065,
#              1392.94,
#              916.408,
#              0.561749,
#              1399.65,
#              877.073,
#              0.571761,
#              1401.57,
#              900.098,
#              0.656358,
#              1403.49,
#              905.854,
#              0.746024,
#              1404.45,
#              907.773,
#              0.520999
#           ]
#        ]
#     }
#  }
    print(openPose)
    # persons=[]: list defined in line 73
    perviousPersons = persons
    persons = []

    # for personData in openPose_out["people"]:
    
    for personData in openPose:
        # persons list, 2 instances of class Person, 'person.Person'
        #  length of list: 2  as 2 people in image
        persons.append(Person(personData))


    # update probability of same person based on distance between keypoints in current & previous frame
    # loop through objects in persons list (2 objects)
    for person in persons:
        
        # draw keypoints and linked keypoints for each person
        person.draw(frame)
        # if previous person exists
        if len(perviousPersons) > 0: 

            # distance(self, other):--> distance 
            # distance between current person
            # distance: obj1_2 with obj1_1,obj2_1; 
            distances = [person.distance(previous) for previous in perviousPersons]
            minDistance = min(distances)
            id = distances.index(minDistance)
            # find same person in previous frame
            previous = perviousPersons[id]


            person.previousFrameDistance = minDistance
            # boundaries to decide it's same same person or not
            if minDistance < 100:
                person.mixInProbabilities(previous, 1)
            if minDistance < 500:
                person.mixInProbabilities(previous, 0.999)
            elif minDistance < 1000:
                person.mixInProbabilities(previous, 0.95)
            elif minDistance < 1200:
                person.mixInProbabilities(previous, 0.7)

            # (left, top, right, bottom) = person.extendedFaceBoundingBox()
            (left, top, right, bottom) = person.faceBoundingBoxFromPose()
            # cv2.FONT_HERSHEY_SIMPLEX: font
            # (int(left),int(bottom)): coordinates of the bottom-left corner of the text string in the image
            # cv2.LINE_AA:lineType
            cv2.putText(frame, str(minDistance),(int(left),int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
    

    # try to recognize faces every 10th frame
    if frames % 10 == 0:

        faces = []
        persons_with_faces = []
        # loop through objects in persons list (2 person.Person objects)
        for person in persons:
            # bigger bounding box (left, top, right, bottom), includes [nose, leye, reye,lear,rear]
            bb = person.faceBoundingBoxFromPose() 
            if bb != None:

                left, top, right, bottom = bb
                faces.append((
                    int(top), 
                    int(right), 
                    int(bottom), 
                    int(left)))
                persons_with_faces.append(person)
        
        # image, faces [(top,right,bottom,left),()]-> encodings of faces
        face_encodings = face_recognition.face_encodings(frame, faces)
        # loop through obj person with faces (0,1)
        for i in range(len(persons_with_faces)):
            person = persons_with_faces[i]
            encoding = face_encodings[i]
            # known faces: encodings of ./student-faces/1/A_0.jpg, A_1.jpg, B_0.jpg
            # compare face encoding with known faces encoding
            distanceToKnown = face_recognition.face_distance(known_faces, encoding)
            # [0.86927221 0.33819645 0.86927221] for known face 1,face1, face2
            # loop2:
            # [0.43819645 0.66927221 0.86927221] for known face 1,face1 ,face2
            # [0.84269277 0.86927221 0.09123462] for known face 1,face1 ,face2

            # lets take best match higher than 0.5
            minDistance = min(distanceToKnown)
            # [0.33819645] for known face 1
            # loop2:
            # [0.43819645] for known face 1
            # [0.09123462] for known face 2
            id = distanceToKnown.tolist().index(minDistance)
            # 1
            # loop2:
            # 0
            # 2

            if minDistance < 0.5:
                
                # detect wrong identification if there was different pose identified strongly with this id:
                # sample faces ids=[0,0,1]
                # id=[0,0,1][1]
                # id=0
                id = ids[id]

                # 被模型预测为正的负样本
                false_positiove = False
                # person ==obj1, other==obj2
                for other in persons:
                    # obj2.idProbs[0]>0.7==> prob obj2 identified as obj1 >0.7
                    if person != other and other.idProbs[id] > 0.7:
                        false_positiove = True
                        break
                # if false positive == false
                if not false_positiove:
                    # update person idProb 
                    person.identify(id, 0.5)
                    left, top, right, bottom = person.faceBoundingBoxFromPose() 

                    # frame + !id:A id:B (person with faces)in white colour
                    cv2.putText(frame, "!id: " + str(names[ids[id]]) , (int(right), int(bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


    # pick unique best matches
    # persons: obj1,obj2
    # p=obj1
    # obj1.idProb[0:2] 
    # probs = [p.idProbs[0:3] for p in persons]
    # probs =[[0.7, 0.3],[0.4,0.6]]
    probs = [p.idProbs[0:2] for p in persons]
    # negativeProbs = [[0.3,0.7],[0.6,0.4]]
    negativeProbs = [[1-p for p in row] for row in probs]

    identified_persons = []

    if len(probs) > 0:
        # found_ids: row_idx[0,1]
        # known_ids: col_idx for min negativeProbs in each row [0,1]
        [found_ids, known_ids] = opt.linear_sum_assignment(negativeProbs)
        # i=0,1
        for i in range(len(known_ids)):
            name = names[known_ids[i]]
            # prob of this person 0.31
            prob = probs[found_ids[i]][known_ids[i]]
            # found_ids=[0,1] 
            # persons[0]-> person = obj1
            person = persons[found_ids[i]]

            # it is very likely the the whole team is captured by the camera
            # to get good result we can use quite low threshold and rely on hungarian algorithm 
            # to uniquely identify students
            if prob > 0.30:
                # obj1.openpose data
                # "person_id":[A],

                person.opData["person_id"] = name

                # "meta":
                #     {"classification": 1 activate/2/3/4
                #     "id_prob": 0.31
                #     "id_probs":[0.31,0.69,]
                #     "previous_frame_distance" : min distance: obj1_2 with obj1_1,obj2_1; 
                #     }
                
                
                person.opData["meta"] = {
                    # data.csv[0][A]  
                    "classification": data[currentDataRow][name],
                    "id_prob" : prob,
                    "id_probs" : person.idProbs,
                    "previous_frame_distance" : person.previousFrameDistance,
                    # gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z
                    "gaze" : gaze_data[currentDataRow][name]
                }
                identified_persons.append(person)
                # faceBoundingBoxFromPose: left, top, right, bottom
                face = person.faceBoundingBoxFromPose()
                if face != None:
                    p_left, p_top, p_right, p_bottom = face
                    # draw rectangle for face bb, colour: aqua
                    cv2.rectangle(frame, (int(p_left),int(p_top)), (int(p_right), int(p_bottom)), (0,255,255), 1)
                    # write yellow text, A prob to frame
                    cv2.putText(frame, name + " " + str(prob),(int(p_left),int(p_top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
    
    # f'./run-result-4/1/number of frames so far/1000
    d = resultFolder + "/" + str(math.floor(frames / 1000))
    Path(d).mkdir(parents=True, exist_ok=True)
    # save image with prob of each face on it in resultFolder
    cv2.imwrite(d + "/{:04d}".format(frames) + "_probabilistic_id.jpg", frame)

    result_data = {}
    
    echo_str = "frame: " + str(frames) + " "


    for p in identified_persons:
        # counts = { "A": 0, "B": 0 }
        # p.opData["person_id"]-> A
        # counts[A] += 1
        # counts = { "A": 1, "B": 0 }
        counts[p.opData["person_id"]] += 1 
        # p.opData["person_id"]-> A
        # result_data = {}
        # result_data = {"A":[A.opData], "B":[B.opData]}
        result_data[p.opData["person_id"]] = p.opData
        # A 0.69 (min distance: obj1_2 with obj1_1,obj2_1)
        echo_str += p.opData["person_id"] + " " + str(p.opData["meta"]["id_prob"]) + " " + str(p.opData["meta"]["previous_frame_distance"]) + " "


    out_data = []
    #     names = [
    #     "A",
    #     "B"]
    for name in names:
        if name in result_data: 
            # result_data = {"A":[A.opData], "B":[B.opData]}
            # p = A.opData
            p = result_data[name]
            # out_data = [1 (active)/2/3/4]
            out_data.append(p["meta"]["classification"])
        else:
            out_data.append(0)

    #     names = [
    #     "A",
    #     "B"]
    for name in names:
        if name in result_data: 
            # result_data = {"A":[A.opData], "B":[B.opData]}
            # p = A.opData
            p = result_data[name]
            # out_data = [4,"pose_keypoints_2d","hand_left_keypoints_2d", "hand_right_keypoints_2d","gaze"]
            out_data.extend(p["pose_keypoints_2d"])
            out_data.extend(p["hand_left_keypoints_2d"])
            out_data.extend(p["hand_right_keypoints_2d"])
            out_data.extend(p["gaze"])
        else:
            # out_data = [0,0*13*+21*3+21*3]
            out_data.extend([0] * (13*3 + 21*3 + 21*3 + 6))

    # out_data = [class,"pose_keypoints_2d","hand_left_keypoints_2d", "hand_right_keypoints_2d"]
    csvWritter.writerow(out_data)
    
    print(echo_str)

    result = {
        "frame": frames,
        # result_data = {"A":[A.opData], "B":[B.opData]}
        "people": result_data
    }
    # ./run-result-id-data/1/'
    Path(data_result_folder).mkdir(parents=True, exist_ok=True)
    # jsonFile = open(resultJson, 'w')
    # resultJson = f'./data/1.json'
    json.dump(result, jsonFile)
    jsonFile.write('\n')

    frames += 1
    snapshots += 1

    if data[currentDataRow]["last_frame"] < frames:
        currentDataRow += 1


print("frames: " + str(frames) + " counts: " + str(counts))

cap.release()
jsonFile.close()
csvFile.close()