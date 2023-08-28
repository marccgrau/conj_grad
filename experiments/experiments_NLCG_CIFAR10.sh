python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data CIFAR10 --dtype float64 --optimizer NLCGEager --batch_size 2500 --gpu 0 --run_eagerly --max_epochs 2 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data CIFAR10 --dtype float64 --optimizer NLCGEager --batch_size 2500 --gpu 0 --run_eagerly --max_epochs 2 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR10 --dtype float64 --optimizer NLCGEager --batch_size 5000 --gpu 0 --run_eagerly --max_epochs 2 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR10 --dtype float64 --optimizer NLCGEager --batch_size 5000 --gpu 0 --run_eagerly --max_epochs 2 --use_wandb

