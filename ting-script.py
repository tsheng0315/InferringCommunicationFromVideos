import json

arr = [
    0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7,
    8, 8, 8,
    9, 9, 9,
    10, 10, 10,
    11, 11, 11,
    12, 12, 12,
    13, 13, 13,
    14, 14, 14,
    15, 15, 15,
    16, 16, 16,
    17, 17, 17,
    18, 18, 18,
    19, 19, 19,
    20, 20, 20,
    21, 21, 21,
    22, 22, 22,
    23, 23, 23,
    24, 24, 24
]

# 9,10,11,12,13,14     19,20,21,22,23,24

def filter(arr):
    return arr[:27] + arr[45:57]




# def process(data):
#     out = {}
#     for person in data["people"]:
#         person_id = person["person_id"][0]
#         if person_id not in person:

#             out[person_id] = {"person_id":[person_id], "pose_keypoints_2d": [], "hand_right_keypoints_2d": [],"hand_left_keypoints_2d": [] }
#         out[person_id]["pose_keypoints_2d"].append(filter(person["pose_keypoints_2d"]))
#         out[person_id]["hand_right_keypoints_2d"].append(person["hand_right_keypoints_2d"])
#         out[person_id]["hand_left_keypoints_2d"].append(person["hand_left_keypoints_2d"])
#     return out

def process(data):
    for person in data["people"]:
        del person["face_keypoints_2d"]
        del person["face_keypoints_3d"]
        del person["pose_keypoints_3d"]
        del person["hand_left_keypoints_3d"]
        del person["hand_right_keypoints_3d"]
        person["pose_keypoints_2d"] = filter(person["pose_keypoints_2d"])
    return data["people"]

with open("imgs0002_keypoints.json") as jsonResource:
    data = json.load(jsonResource)
    print(json.dumps(process(data)))

 