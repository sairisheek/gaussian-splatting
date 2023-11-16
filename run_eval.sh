#CUDA_VISIBLE_DEVICES=0 python train.py -s $1 -m $2 --checkpoint_iterations 30000 --prune_sched 15000 20000 25000 --densify_lag 200 --lambda_depth 0.05
CUDA_VISIBLE_DEVICES=0 python render.py --no_load_depth -s $1 -m $2 --iteration 30000
python metrics.py -m $2 -e $3