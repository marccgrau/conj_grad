# NLCG Optimization for Deep Learning

This is the base repository for the Master's thesis in Computer Science at the University of St. Gallen.
To start training run the script `main.py` with all necessary arguments. These include:

-   path
    -   Directory to where data should be stored
-   model
    -   Choose which model should be used for training, they should be defined in src/models/model_archs
-   model_size
    -   Depending on the model will allow to determine its size
-   data
    -   Choose the dataset for optimization (MNIST, Fashion MNIST, CIFAR10, CIFAR100, SVHN)
-   dtype
    -   Precision used (float32, float64)
-   optimizer
    -   Decide on the optimizer (NLCG, Adam, RMSProp, SGD)
-   batch_size
    -   Choose the size of your batches
-   gpu
    -   In case of multiple available GPUs, decide how many and which ones you want to use
-   run_eagerly
    -   Use eager execution in TensorFlow
-   max_epochs
    -   Limit the number of epochs
-   use_wandb
    -   Track your experiments with WandB
-   max_calls
    -   Limit the number of optimization steps

Here is an exemplary run for the script:

`python main.py --path "/home/user/code/data/tf_data" --model FlatMLP --model_size small --data MNIST --dtype float64 --optimizer ADAM --batch_size 6000 --gpu 4 --run_eagerly --max_epochs 50000 --use_wandb --max_calls 6697`

Find more in the experiment folder.
