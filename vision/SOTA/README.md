### Making directory for CIFAR-10 dataset
     mkdir dataset

### Download 10 trained models on each class of CIFAR-10
https://drive.google.com/file/d/1P2JJAj-ZhKi_z4jiTY6lWhS8cx8bGDpC/view?usp=sharing
    
    mv saved_models.zip .
    unzip saved_models.zip -d saved_models

### OOD detection (SOTA, Table 2 results on CIFAR-10)

    For ICAD - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu 0 --n 1 --indist_class 10 --proper_train_size 4500 --trials 5
    For ours - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu 0 --n 5 --indist_class 10 --proper_train_size 4500 --trials 5
    
### Optional - Train models on single class of CIFAR-10
    python train.py --cuda --outf saved_models --dataroot dataset --gpu $gpu_num$ --class_num $0-9$ --niter 200 --train_size 4500 --archi_type wrn
 
