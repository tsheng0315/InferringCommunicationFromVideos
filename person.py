import sys
import cv2
import math

BODY_NOSE = 0
BODY_NECT = 1
BODY_RIGHT_EYE = 9
BODY_LEFT_EYE = 10
BODY_RIGHT_EAR = 11
BODY_LEFT_EAR = 12

# Taken from OpenPose source code for reference (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-cpython)
# Result for BODY_25 (25 body parts consisting of COCO + foot)
# const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
#     {0,  "Nose"},
#     {1,  "Neck"},
#     {2,  "RShoulder"},
#     {3,  "RElbow"},
#     {4,  "RWrist"},
#     {5,  "LShoulder"},
#     {6,  "LElbow"},
#     {7,  "LWrist"},
#     {8,  "MidHp"},
#     {9,  "RHip"},
#     {10, "RKnee"},
#     {11, "RAnkle"},
#     {12, "LHip"},
#     {13, "LKnee"},
#     {14, "LAnkle"},
#     {15, "REye"},
#     {16, "LEye"},
#     {17, "REar"},
#     {18, "LEar"},
#     {19, "LBigToe"},
#     {20, "LSmallToe"},
#     {21, "LHeel"},
#     {22, "RBigToe"},
#     {23, "RSmallToe"},
#     {24, "RHeel"},
#     {25, "Background"}

# my partial openpose
#     {0,  "Nose"},
#     {1,  "Neck"},
#     {2,  "RShoulder"},
#     {3,  "RElbow"},
#     {4,  "RWrist"},
#     {5,  "LShoulder"},
#     {6,  "LElbow"},
#     {7,  "LWrist"},
#     {8,  "MidHip"},
#     15-->{9, "REye"},
#     16-->{10, "LEye"},
#     17-->{11, "REar"},
#     18-->{12, "LEar"},
#     


# BODY_PAIRS = [
#     (17,15),
#     (15,0), 
#     (0,16), 
#     (16,18),

#     (0,1),

#     (4,3),
#     (3,2),
#     (2,1),

#     (1,5),
#     (5,6),
#     (6,7),

#     (1,8),
#     # (8,9),
#     # (9,10),  
#     # (10,11),
#     # (11,24),
#     # (11,22),
#     # (22,23),

#     (1,2),
#     (2,3),
#     (3,4), 

#     (1,5), 
#     (5,6),
#     (6,7),  
# ]

BODY_PAIRS = [
    (11,9),
    (9,0), 
    (0,10), 
    (10,12),

    (0,1),

    (4,3),
    (3,2),
    (2,1),

    (1,5),
    (5,6),
    (6,7),

    (1,8),

    (1,2),
    (2,3),
    (3,4), 

    (1,5), 
    (5,6),
    (6,7),  
]



class Person:
    def __init__(self, opData):
        # openpose data
        self.opData = opData
        # 
        # self.idProbs = [1,1,1,1]
        self.idProbs = [1,1,1]
        self.normaliseProbs()
        self.previousFrameDistance = math.inf
    

    # normalise idProbs 
    def normaliseProbs(self):
        s = sum(self.idProbs)
        # for i in range(len(self.idProbs)):
        for i, _ in enumerate(self.idProbs):
            self.idProbs[i] /= s
    

    # get pose point from openpose output (x,y confidence) 
    def getPosePoint(self, data, index):
        return (data[index * 3], data[index * 3 + 1], data[index * 3 + 2])

    # make sure bounding box includes keypoints from opnepose
    # points: keypoints from openpose
    def includePoints(self, bb, points):
        left, top, right, bottom = bb

        for x, y, c in points:
            if c > 0: 
                top = min(top, y)
                bottom = max(bottom, y)
                left = min(left, x)
                right = max(right, x)

        return (left, top, right, bottom)


    # empty bounding box
    def emptyBB(self):
        top = math.inf
        bottom = 0
        left = math.inf
        right = 0
        return (left, top, right, bottom)
    # return T/F
    def isEmpty(self, bb):
        left, top, right, bottom = bb
        return top == math.inf and bottom == 0 and left == math.inf and right == 0
    

    # make bigger boundingbox  
    def extendBox(self, bb, t, r, b, l):
        left, top, right, bottom = bb

        height = bottom - top
        width = right - left

        top = top - t * height
        right = right + r * width
        bottom = bottom + b * height
        left = left - l * width

        return (left, top, right, bottom)


    def faceBoundingBoxFromPose(self):
        body = self.opData["pose_keypoints_2d"]
        nose = self.getPosePoint(body, BODY_NOSE)
        # (data[0], data[1], data[2])
        leye = self.getPosePoint(body, BODY_LEFT_EYE)
        reye = self.getPosePoint(body, BODY_RIGHT_EYE)
        lear = self.getPosePoint(body, BODY_LEFT_EAR)
        rear = self.getPosePoint(body, BODY_RIGHT_EAR)
        # init bounding box is empty
        top = math.inf
        bottom = 0
        left = math.inf
        right = 0

        # def includePoints(self, bb, points)--> bigger (left, top, right, bottom)
        # emptyBB()--> (left, top, right, bottom)
        center = self.includePoints(self.emptyBB(), [nose, leye, reye])
        # def isEmpty(self, bb)--> T/F
        # if the bigger bounding box is still empty, return None and quit function
        if self.isEmpty(center):
            return None

        # def extendBox():-->bigger (left, top, right, bottom) 
        # if the bigger bounding box is not empty, make it larger
        bb = self.extendBox(center, t=1, r=0.5, b=2.5, l=0.5)
        # def includePoints(self, bb, points)--> bigger (left, top, right, bottom)
        return  self.includePoints(bb, [lear, rear])
 
    # crushed
    # def faceBoundingBox(self):
    #     # print(self.opData["face_keypoints_2d"])
    #     face = self.opData["face_keypoints_2d"]
       
    #     top = math.inf
    #     bottom = 0
    #     left = math.inf
    #     right = 0

    #     # find (left, top, right, bottom) for face bounding box
    #     it = iter(face)
    #     for x in it:
    #         y = next(it)
    #         next(it)
    #         top = min(top, y)
    #         bottom = max(bottom, y)
    #         left = min(left, x)
    #         right = max(right, x)
    #     return (left, top, right, bottom)
    
    # def extendedFaceBoundingBox(self):
    #     (left, top, right, bottom) = self.faceBoundingBox()
    #     height = bottom - top
    #     width = right - left

    #     if width <= 2:
    #         return (0,0,0,0)
    #     top = top - 0.3 * height
    #     right = right + 0.1 * width
    #     bottom = bottom + 0.1 * height
    #     left = left - 0.1 * width
    #     return (left, top, right, bottom)

    def distance(self, other):
        """Calculate Distance between 2 persons as sum of open pose keypoint distances"""

        MISSING_KEYPOINT_DISTANCE = 50
        # person 1 openpose keypoint
        keypoints = self.opData["pose_keypoints_2d"]
        # person 2 openpose keypoint
        otherKeypoints = other.opData["pose_keypoints_2d"]
        
        distance = 0
        it = iter(zip(keypoints, otherKeypoints))
        for (x1,x2) in it:
            (y1, y2) = next(it)
            # if one person not exists
            if ((x1 == 0 and y1 == 0 and x2 != 0 and y2 != 0) or
                (x1 != 0 and y1 != 0 and x2 == 0 and y2 == 0)):
                distance += MISSING_KEYPOINT_DISTANCE
            else:
                distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # skip c
            next(it)
        return distance
            
    #  updates idProbs of each person
    def identify(self, id, probability):
        # in case of fail, give equal chance of other ids
        # self.idProbs = [1,1,1]
        # probability of wrong identify
        probabilityOfWrong = (1 - probability) / (len(self.idProbs) - 1)
        # range starts from 0,1,2
        for i in range(len(self.idProbs)):
            if i == id:
                # i=0: [0.5,1,1]
                self.idProbs[i] *= probability
            else:
                # i=1: [0.5,0.16,1]
                # i=2:[0.5,0.16,0.16]
                self.idProbs[i] *= probabilityOfWrong
        self.normaliseProbs()

    #  p=previous person, probabilitySame= probability of this is the same person A as in the previous frame
    def mixInProbabilities(self, p, probabilitySame):
        self.idProbs = [me * (1 - probabilitySame) + other * probabilitySame for (me, other) in zip(self.idProbs, p.idProbs)]

    # draw circle and lines between linked keypoints on image
    def draw(self, frame):
        pose = self.opData["pose_keypoints_2d"]
        for i1,i2 in BODY_PAIRS:
            x1 = pose[i1 * 3]
            y1 = pose[i1 * 3 + 1]
            x2 = pose[i2 * 3]
            y2 = pose[i2 * 3 + 1]
            cv2.circle(frame, (int(x1), int(y1)), 2, (0,0,255), 1)

            if (x1 != 0 or y1 != 0) and (x2 != 0 or y2 != 0):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1)
