python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer SGD --batch_size 36 --gpu 9 --run_eagerly --max_epochs 6 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer SGD --batch_size 36 --gpu 9 --run_eagerly --max_epochs 7 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data MNIST --dtype float64 --optimizer SGD --batch_size 36 --gpu 9 --run_eagerly --max_epochs 6 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data MNIST --dtype float64 --optimizer SGD --batch_size 36 --gpu 9 --run_eagerly --max_epochs 7 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data FASHION_MNIST --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data FASHION_MNIST --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data FASHION_MNIST --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data FASHION_MNIST --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data CIFAR10 --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data CIFAR10 --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR10 --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR10 --dtype float64 --optimizer SGD --batch_size 35 --gpu 9 --run_eagerly --max_epochs 10 --use_wandb

