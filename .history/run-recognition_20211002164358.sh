
for VIDEO in 1 # 2 3 4 5 6 7
do
    echo 
    echo $VIDEO
    python3 mark-faces.py $VIDEO > ./data/$VIDEO.log
done