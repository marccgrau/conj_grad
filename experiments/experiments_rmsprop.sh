python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 50 --gpu 11 --run_eagerly --max_epochs 3 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 50 --gpu 11 --run_eagerly --max_epochs 3 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 50 --gpu 11 --run_eagerly --max_epochs 3 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 50 --gpu 11 --run_eagerly --max_epochs 3 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 15 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 15 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 7 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 7 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 20 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 20 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 7 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 11 --run_eagerly --max_epochs 7 --use_wandb

