python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size small --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 9 --run_eagerly --max_epochs 43 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size large --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 9 --run_eagerly --max_epochs 38 --use_wandb

#python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 9 --run_eagerly --max_epochs 24 --use_wandb

#python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 75 --gpu 9 --run_eagerly --max_epochs 24 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size small --data CIFAR100 --dtype float64 --optimizer ADAM --batch_size 75 --gpu 9 --run_eagerly --max_epochs 43 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size large --data CIFAR100 --dtype float64 --optimizer ADAM --batch_size 75 --gpu 9 --run_eagerly --max_epochs 38 --use_wandb

#python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR100 --dtype float64 --optimizer ADAM --batch_size 75 --gpu 9 --run_eagerly --max_epochs 24 --use_wandb

#python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR100 --dtype float64 --optimizer ADAM --batch_size 75 --gpu 9 --run_eagerly --max_epochs 24 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size small --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 75 --gpu 9 --run_eagerly --max_epochs 43 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size large --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 75 --gpu 9 --run_eagerly --max_epochs 38 --use_wandb

#python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 75 --gpu 9 --run_eagerly --max_epochs 24 --use_wandb

#python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 75 --gpu 9 --run_eagerly --max_epochs 24 --use_wandb


