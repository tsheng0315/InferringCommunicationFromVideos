$ docker run -it --rm algebr/openface:latest


# open a second shell
$ docker cp /home/ting/Downloads/video2frames/xxxxxxx0001.jpg ca93f1e07718:/home/openface-build


# in first shell
$ build/bin/FaceLandmarkImg -f sample1.jpg