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
python train.py -s data/bunny_hallway -m output/bunny_hallway_full  --checkpoint_iterations 30000
python train.py -s data/bunny_hallway -m output/bunny_hallway_40 --step 1 --max_cameras 40 --checkpoint_iterations 30000
python train.py -s data/bunny_hallway -m output/bunny_hallway_30 --step 3 --max_cameras 30 --checkpoint_iterations 30000
python train.py -s data/bunny_hallway -m output/bunny_hallway_20  --step 2 --max_cameras 20 --checkpoint_iterations 30000
python train.py -s data/bunny_hallway -m output/bunny_hallway_15  --step 3 --max_cameras 15 --checkpoint_iterations 30000
python train.py -s data/bunny_hallway -m output/bunny_hallway_10  --step 4 --max_cameras 10 --checkpoint_iterations 30000

python train.py -s data/bunny_vmglab -m output/bunny_vmglab_full  --checkpoint_iterations 30000
python train.py -s data/bunny_vmglab -m output/bunny_vmglab_40 --step 1 --max_cameras 40 --checkpoint_iterations 30000
python train.py -s data/bunny_vmglab -m output/bunny_vmglab_30 --step 3 --max_cameras 30 --checkpoint_iterations 30000
python train.py -s data/bunny_vmglab -m output/bunny_vmglab_20  --step 2 --max_cameras 20 --checkpoint_iterations 30000
python train.py -s data/bunny_vmglab -m output/bunny_vmglab_15  --step 3 --max_cameras 15 --checkpoint_iterations 30000
python train.py -s data/bunny_vmglab -m output/bunny_vmglab_10  --step 4 --max_cameras 10 --checkpoint_iterations 30000


