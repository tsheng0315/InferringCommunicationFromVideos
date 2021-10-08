# limit number of used GPU devices
export CUDA_VISIBLE_DEVICES=1

for VIDEO in  1 #2 3 4 5 6 7
do
    RESULTS=./run-result-hands
    # VIDEO_RESULT=$RESULTS/$VIDEO

    echo ------------------
    echo running: $VIDEO
    echo

    mkdir -p $RESULTS/$VIDEO
    ./openpose/build/examples/openpose/openpose.bin \
        --model_folder ./openpose/models/ \
        --video ./videosLabelled/$VIDEO/webcam.mp4 \
        --write_json=$RESULTS/$VIDEO \
        --display=0 \
        --render_pose 1 \
        --model_pose BODY_25 \
        --cli_verbose=1 \
        --hand
done