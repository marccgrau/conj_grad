python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data FASHION_MNIST --dtype float64 --optimizer NLCGEager --batch_size 2000 --gpu 2 --run_eagerly --max_epochs 5 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data FASHION_MNIST --dtype float64 --optimizer NLCGEager --batch_size 2000 --gpu 2 --run_eagerly --max_epochs 5 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data FASHION_MNIST --dtype float64 --optimizer NLCGEager --batch_size 2000 --gpu 2 --run_eagerly --max_epochs 5 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data FASHION_MNIST --dtype float64 --optimizer NLCGEager --batch_size 2000 --gpu 2 --run_eagerly --max_epochs 5 --use_wandb
