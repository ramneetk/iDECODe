### Making directory for CIFAR10 dataset
     mkdir dataset

### Trained models  
    mv ../../../saved_models.zip .
    unzip saved_models.zip -d saved_models

### OOD detection (SOTA, Table 2 results)

    For ICAD - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu 0 --n 1 --indist_class 10 --proper_train_size 4500 --trials 5
    For ours - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu 0 --n 5 --indist_class 10 --proper_train_size 4500 --trials 5
 
