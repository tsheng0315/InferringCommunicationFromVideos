import sys
import cv2
import face_recognition
from pathlib import Path

# video_id = sys.argv[1]
# video = f'./videosLabelled/{video_id}/webcam.mp4'
# resultFolder = f'./run-result-face-images/{video_id}/'

video = f'./videosLabelled/1/webcam.mp4'
resultFolder = f'./run-result-face-images/1/'

Path(resultFolder).mkdir(parents=True, exist_ok=True)

# create video capture object
cap = cv2.VideoCapture(video)
# get frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

frames = 0
snapshots = 0
snapshotSeconds = 1.0

# frame and snapshot are basiccally the same, but 
while(cap.isOpened()):
    
    ret, frame = cap.read()
    frames += 1
    videoTime = frames / fps
    # if doesn't have a next frame--> break
    if ret == False:
        break
    # for every 10 snapshots
    if snapshots % 10 == 0:
        print("time ", videoTime)
        cv2.imwrite(resultFolder + str(snapshots) + "_img.jpg", frame)

        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
        print(face_locations)
        
        faceno = 0
        # loop through location of faces 
        for face_location in face_locations:

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            top = max(0, top - 20)
            left = max(0, left - 20)
            bottom = bottom + 20
            right = right + 20

            image = frame[top:bottom, left:right]

            if len(face_recognition.face_encodings(image)) > 0:
                cv2.imwrite(resultFolder + str(snapshots) + "_face_" + str(faceno) + ".jpg", image)
                faceno += 1


    snapshots += 1

    # skip frames until 
    while (cap.isOpened() and videoTime < snapshots * snapshotSeconds):
        cap.read()
        frames += 1
        videoTime = frames / fps

cap.release()
cv2.destroyAllWindows()