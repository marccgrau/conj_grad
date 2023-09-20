python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data SVHN --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 3 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 25661

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data SVHN --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 3 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 24767

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data SVHN --dtype float64 --optimizer RMSPROP --batch_size 5000 --gpu 3 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 7724

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data SVHN --dtype float64 --optimizer RMSPROP --batch_size 5000 --gpu 3 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 8260





python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data SVHN --dtype float64 --optimizer SGD --batch_size 2500 --gpu 5 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 25661

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data SVHN --dtype float64 --optimizer SGD --batch_size 2500 --gpu 5 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 24767

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data SVHN --dtype float64 --optimizer SGD --batch_size 5000 --gpu 5 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 7724

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data SVHN --dtype float64 --optimizer SGD --batch_size 5000 --gpu 8 --run_eagerly --max_epochs 500000 --use_wandb --max_calls 8260


