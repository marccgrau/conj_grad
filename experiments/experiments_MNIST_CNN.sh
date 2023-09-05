python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 50 --gpu 11 --run_eagerly --max_epochs 4 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer RMSPROP --batch_size 50 --gpu 11 --run_eagerly --max_epochs 4 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer ADAM --batch_size 50 --gpu 10 --run_eagerly --max_epochs 4 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer ADAM --batch_size 50 --gpu 10 --run_eagerly --max_epochs 4 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer SGD --batch_size 50 --gpu 9 --run_eagerly --max_epochs 4 --use_wandb

python main.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer SGD --batch_size 50 --gpu 9 --run_eagerly --max_epochs 4 --use_wandb



