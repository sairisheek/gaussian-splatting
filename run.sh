#python train.py -s data/clutter -m output/clutter_5_3 --step 3 --max_cameras 5 --checkpoint_iterations 30000
#python train.py -s data/clutter -m output/clutter_10_2 --step 2 --max_cameras 10 --checkpoint_iterations 30000
#python train.py -s data/clutter -m output/clutter_15 --step 2 --max_cameras 15 --checkpoint_iterations 30000
#python train.py -s data/clutter -m output/clutter_20 --step 2 --max_cameras 20 --checkpoint_iterations 30000
#python train.py -s data/clutter -m output/clutter_30 --step 2 --max_cameras 30 --checkpoint_iterations 30000
#python train.py -s data/clutter -m output/clutter_40 --step 2 --max_cameras 40 --checkpoint_iterations 30000
#python train.py -s data/clutter -m output/clutter_50 --step 2 --max_cameras 50 --checkpoint_iterations 30000
#python train.py -s data/far_20 -m output/far_20_01 --lambda_depth 0.1 --checkpoint_iterations 30000
#python train.py -s data/far_20 -m output/far_20_02 --lambda_depth 0.2 --checkpoint_iterations 30000
#python train.py -s data/far_20 -m output/far_20_05 --lambda_depth 0.5 --checkpoint_iterations 30000
#python train.py -s data/far_20 -m output/far_20_1 --lambda_depth 1 --checkpoint_iterations 30000
#python train.py -s data/bunny_hallway -m output/bunny_hallway_full_pearson --lambda_depth 0.05 --checkpoint_iterations 30000
#python train.py -s data/bunny_hallway -m output/bunny_hallway_40_pearson --lambda_depth 0.05 --step 1 --max_cameras 40 --checkpoint_iterations 30000
#python train.py -s data/bunny_hallway -m output/bunny_hallway_30_pearson --lambda_depth 0.05 --step 3 --max_cameras 30 --checkpoint_iterations 30000
#python train.py -s data/bunny_hallway -m output/bunny_hallway_20_pearson --lambda_depth 0.05 --step 2 --max_cameras 20 --checkpoint_iterations 30000
#python train.py -s data/bunny_hallway -m output/bunny_hallway_15_pearson --lambda_depth 0.05 --step 3 --max_cameras 15 --checkpoint_iterations 30000
#python train.py -s data/bunny_hallway -m output/bunny_hallway_10_pearson --lambda_depth 0.05 --step 4 --max_cameras 10 --checkpoint_iterations 30000

#python train.py -s data/bunny_vmglab -m output/bunny_vmglab_full_pearson --lambda_depth 0.05 --checkpoint_iterations 30000
#python train.py -s data/bunny_vmglab -m output/bunny_vmglab_40_pearson --lambda_depth 0.05 --step 1 --max_cameras 40 --checkpoint_iterations 30000
#python train.py -s data/bunny_vmglab -m output/bunny_vmglab_30_pearson --lambda_depth 0.05 --step 3 --max_cameras 30 --checkpoint_iterations 30000
#python train.py -s data/bunny_vmglab -m output/bunny_vmglab_20_pearson --lambda_depth 0.05 --step 2 --max_cameras 20 --checkpoint_iterations 30000
#python train.py -s data/bunny_vmglab -m output/bunny_vmglab_15_pearson --lambda_depth 0.05 --step 3 --max_cameras 15 --checkpoint_iterations 30000
#python train.py -s data/bunny_vmglab -m output/bunny_vmglab_10_pearson --lambda_depth 0.05 --step 4 --max_cameras 10 --checkpoint_iterations 30000
# CUDA_VISIBLE_DEVICES=1 python train.py -s data/garden_12 -m output/test1 --lambda_ranking 0.05 --box_s 300 --n_corr 5000
# CUDA_VISIBLE_DEVICES=1 python train.py -s data/garden_12 -m output/test2 --lambda_ranking 0.05 --box_s 300 --n_corr 10000
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/garden_12 -m output/avg_depth_05 --lambda_depth 0.05 --detect_anomaly
#CUDA_VISIBLE_DEVICES=1 python train.py -s data/garden_12 -m output/garden_12_test2_avg_1 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --iterations 35000 --beta 0.1 --save_iterations 35000 
CUDA_VISIBLE_DEVICES=0 python train.py -s data/garden_12 -m output/garden_12_prune_dense_1 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 2600 --prune_dense_interval 500 &

CUDA_VISIBLE_DEVICES=1 python train.py -s data/garden_12 -m output/garden_12_prune_dense_2 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 2000 --prune_dense_interval 500 &

CUDA_VISIBLE_DEVICES=2 python train.py -s data/garden_12 -m output/garden_12_prune_dense_3 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 1300 --prune_dense_interval 500 

# CUDA_VISIBLE_DEVICES=2 python train.py -s data/garden_12 -m output/garden_12_prune_dense_4 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 2600 --prune_dense_interval 1000 

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py -s ../gaussian-splatting/data/bunny_hallway -m output/bunny_depth_rank_64 --lambda_rank 0.1 --box_s 64 --n_corr 800 --max_cameras 10 --step 4




