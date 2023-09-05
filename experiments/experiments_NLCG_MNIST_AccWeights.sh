python main_acc_weights.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data MNIST --dtype float64 --optimizer NLCGAccWeights --batch_size 6000 --gpu 3 --run_eagerly --max_epochs 2 --use_wandb

python main_acc_weights.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size large --data MNIST --dtype float64 --optimizer NLCGAccWeights --batch_size 6000 --gpu 3 --run_eagerly --max_epochs 2 --use_wandb

python main_acc_weights.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size small --data MNIST --dtype float64 --optimizer NLCGAccWeights --batch_size 4000 --gpu 3 --run_eagerly --max_epochs 2 --use_wandb

python main_acc_weights.py --path "/home/user/code/data/tf_data" --model FlatCNN --model_size large --data MNIST --dtype float64 --optimizer NLCGAccWeights --batch_size 4000 --gpu 3 --run_eagerly --max_epochs 2 --use_wandb




