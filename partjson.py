
import json
with open("openposepart.json") as jsonResource:
    data = json.load(jsonResource)
    for idx in data:
        print(idx)



# with open("imgs0002_keypoints.json") as jsonResource:
#     data = json.load(jsonResource)
#     for person in data['people']:
#         print(person)