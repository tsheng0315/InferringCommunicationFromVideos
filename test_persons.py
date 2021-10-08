# import json
# from person import Person
# import cv2

# with open(f'./run-result-hands/1/' + "imgs0002_keypoints.json") as file:
#         openPose = json.load(file)
# # perviousPersons = persons
# persons = []

# for personData in openPose["people"]:
#     persons.append(Person(personData))
#     print(type(Person(personData)))


import json
from person import Person
import cv2

with open(f'./run-result-hands/1/' + "imgs0002_keypoints.json") as file:
        openPose = json.load(file)
# perviousPersons = persons
persons = []

print(json.dumps(openPose['people'].values()))

for personData in openPose['people']:
    print(personData)
    
