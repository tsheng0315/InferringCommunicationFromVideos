
MODEL=TeamModel2
MODEL_FOLDER=results/team-2

CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-0 $MODEL 0 
wait
CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-1 $MODEL 1 
wait
CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-2 $MODEL 2 
wait

CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-3 $MODEL 3 
wait 

CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-4 $MODEL 4 
wait 

CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-5 $MODEL 5 
wait 

CUDA_VISIBLE_DEVICES=0 python3 ./classifier/learn.py $MODEL_FOLDER/fold-6 $MODEL 6 
wait