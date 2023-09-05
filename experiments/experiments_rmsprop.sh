python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 3000 --gpu 11 --run_eagerly --max_epochs 3 --use_wandb --max_calls 10985

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 3000 --gpu 11 --run_eagerly --max_epochs 3 --use_wandb --max_calls 10985

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 6000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 7545

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 6000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 6426

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 3000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 25293

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 3000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 23236

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 6000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 9519

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data FASHION_MNIST --dtype float64 --optimizer RMSPROP --batch_size 6000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 9500

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 26454

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 2500 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 23651

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 5000 --gpu 11 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 9452

python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data CIFAR10 --dtype float64 --optimizer RMSPROP --batch_size 5000 --gpu 11 --run_eagerly --max_epochs 50000  --use_wandb --max_calls 8094