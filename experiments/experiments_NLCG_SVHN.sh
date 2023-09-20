python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data SVHN --dtype float64 --optimizer NLCGEager --batch_size 5000 --gpu 4 --run_eagerly --max_epochs 3 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data SVHN --dtype float64 --optimizer NLCGEager --batch_size 5000 --gpu 5 --run_eagerly --max_epochs 3 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data SVHN --dtype float64 --optimizer NLCGEager --batch_size 2500 --gpu 6 --run_eagerly --max_epochs 2 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data SVHN --dtype float64 --optimizer NLCGEager --batch_size 2500 --gpu 8 --run_eagerly --max_epochs 2 --use_wandb

