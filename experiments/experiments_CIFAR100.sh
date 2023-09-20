python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size small --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 6 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 30171

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size large --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 7 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 26182

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 8 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 22252

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR100 --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 11 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 21678





python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size small --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 2500 --gpu 8 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 30171

python main.py --path "/home/user/code/data/tf_data" --model FlatCNNCifar100 --model_size large --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 2500 --gpu 8 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 26182

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 2500 --gpu 8 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 22252

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR100 --dtype float64 --optimizer SGD --batch_size 2500 --gpu 8 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 21678


