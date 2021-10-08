$ docker run -it --rm algebr/openface:latest

$ docker cp samples/sample1.jpg 3a73fbce562e:/home/openface-build

$ build/bin/FaceLandmarkImg -f sample1.jpg