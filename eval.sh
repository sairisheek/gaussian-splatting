CUDA_VISIBLE_DEVICES=1 python render.py --no_load_depth -s $1 -m $2 --iteration 30000
python metrics.py -m $2 -e $3