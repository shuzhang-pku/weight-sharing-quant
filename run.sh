python -m torch.distributed.launch --nproc_per_node 8 eval_cifar100.py --base 8
python -m torch.distributed.launch --nproc_per_node 8 eval_cifar100.py --base 16
