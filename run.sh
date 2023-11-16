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
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/garden_12 -m output/garden_12_prune_dense_1 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 2600 --prune_dense_interval 500 &

#CUDA_VISIBLE_DEVICES=1 python train.py -s data/garden_12 -m output/garden_12_prune_dense_2 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 2000 --prune_dense_interval 500 &

#CUDA_VISIBLE_DEVICES=2 python train.py -s data/garden_12 -m output/garden_12_prune_dense_3 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 1300 --prune_dense_interval 500 

# CUDA_VISIBLE_DEVICES=2 python train.py -s data/garden_12 -m output/garden_12_prune_dense_4 --lambda_depth 0.05 --start_checkpoint output/garden_12_test2/chkpnt30000.pth --prune_thresh 0.05 --iterations 35000 --save_iterations 35000 --prune_interval 2600 --prune_dense_interval 1000 

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py -s ../gaussian-splatting/data/bunny_hallway -m output/bunny_depth_rank_64 --lambda_rank 0.1 --box_s 64 --n_corr 800 --max_cameras 10 --step 4

# CUDA_VISIBLE_DEVICES=0 python train.py -s data/garden_12_f -m output/garden_12_prune_stop --prune_sched 15000 20000 25000 --densify_lag 200 --lambda_depth 0.05 --pixel_thresh 0.001 --prune_stop 0.1 --thresh_bin 0.05 --start_checkpoint output/garden_12_prune_stop/chkpnt14000.pth
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020 
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 99 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20000 --front_mask --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 20000 --front_mask --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 99 --prune_sched 20000 --front_mask --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump -m output/stump_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump -m output/stump_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump -m output/stump_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 99 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump -m output/stump_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20000 --front_mask --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump -m output/stump_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 20000 --front_mask --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 20020
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump -m output/stump_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 20020

#CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m thresh_output/bicycle_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20000 --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 20020 
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/counter_12_f -m output/counter_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 20010 --power_thresh -4.0 --start_checkpoint output/counter_12_local_pearson/chkpnt20000.pth --iterations 20020
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/counter_12_f -m output/counter_12_prune_minmax --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 20010 --power_thresh -4.0 --start_checkpoint output/counter_12_local_pearson/chkpnt20000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/bicycle_12_f -m thresh_output/bicycle_12_prune_norm --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 96 --prune_sched 19010 --power_thresh -4.0 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth --iterations 19020 
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/counter_12_f -m thresh_output/counter_12_prune_norm --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 99 --prune_sched 20010 --power_thresh -4.0 --start_checkpoint output/counter_12_local_pearson/chkpnt20000.pth --iterations 20020
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump_12_f -m output/stump_12_prune_norm --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 19010 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 19020
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/kitchen_12_f -m output/kitchen_12_prune_norm --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 19950 --power_thresh -4.0 --start_checkpoint output/kitchen_12_local_pearson/chkpnt19900.pth --iterations 20000

#CUDA_VISIBLE_DEVICES=0 python train.py -s data/bicycle_12_f  -m final_output/bicycle_12_prune_test --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_sched 19010 --power_thresh -4.0 --densify_lag 200 --densify_period 5000 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth 
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/stump_12_f  -m output_thresh/stump_12_prune_dip --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 95 --prune_sched 19010 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 19020
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/kitchen_12_f -m output/kitchen_12_prune_test --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_sched 19950 25000 --power_thresh -4.0 --start_checkpoint output/kitchen_12_local_pearson/chkpnt19900.pth --densify_lag 200 --densify_period 5000
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/garden_12_f -m thresh_output/garden_12_prune_dip2 --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 19950 --power_thresh -4.0 --start_checkpoint output/garden_12_lcoal_pearson/chkpnt19900.pth --iterations 20000
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/stump_12_f  -m thresh_output/stump_12_prune_dip2 --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 90 --prune_sched 19010 --power_thresh -4.0 --start_checkpoint output/stump_12_local_pearson/chkpnt19000.pth --iterations 19020
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/counter_12_f -m thresh_output/counter_12_prune_dip2 --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 20010 --power_thresh -4.0 --start_checkpoint output/counter_12_local_pearson/chkpnt20000.pth --iterations 20020
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/bonsai_12_f -m thresh_output/bonsai_12_prune_dip2 -lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_percentile 97 --prune_sched 19010 --start_checkpoint output/bonsai_12_local_pearson/chkpnt19000.pth --iterations 19020
#CUDA_VISIBLE_DEVICES=0 python train.py -s data/bicycle_12_f  -m final_output/bicycle_12_prune_test2 --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_sched 19010 24000 --power_thresh -4.0 --densify_lag 500 --densify_period 1000 --start_checkpoint output/bicycle_12_local_pearson/chkpnt19000.pth 
#CUDA_VISIBLE_DEVICES=2 python train.py -s data/garden_12_f  -m final_output/garden_12_prune_test2 --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_sched 19910 25000 --power_thresh -4.0 --densify_lag 500 --densify_period 1000 --start_checkpoint output/garden_12_local_pearson/chkpnt19900.pth  
CUDA_VISIBLE_DEVICES=0 python train.py -s data/stump_12_f  -m paper_output/stump_12_test --lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_sched 20000 25000 --power_thresh -4.0 --densify_period 1000000 --lambda_diffusion 0.0005 --step_ratio 0.99 
CUDA_VISIBLE_DEVICES=2 python train.py -s data/bonsai_12_f -m final_output/bonsai_12_prune_test5 -lambda_local_pearson 0.1 --box_p 128 --p_corr 0.5 --prune_sched 19010 25000 --start_checkpoint output/bonsai_12_local_pearson/chkpnt19000.pth --densify_lag 50000 --power_thresh -4.0
