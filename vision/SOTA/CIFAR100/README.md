### Making directory for CIFAR-100 dataset and trained models
     mkdir dataset
     mkdir models

### Download 10 trained models on each class of CIFAR-100
https://drive.google.com/drive/folders/1pcpFK7244kFhu4RryAlSHNa1hta86rW9?usp=sharing
    
    mv saved_models.zip .
    unzip saved_models.zip -d saved_models

### OOD detection (SOTA, Table 7 results on CIFAR-100)
#### For getting results on all 20 classes
    For ICAD - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu $0/1/2/3$ --n 1 --indist_class 20 --proper_train_size 2000 --trials 5 --archi_type resnet18
    For ours - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu $0/1/2/3$ --n 5 --indist_class 20 --proper_train_size 2000 --trials 5 --archi_type resnet18
#### For getting results on 1 class, ex. class 0
   For ICAD - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu $0/1/2/3$ --n 1 --net saved_models/class0.pth --ood_dataset cifar_non0_class  --indist_class 0 --proper_train_size 2000 --trials 5 --archi_type resnet18
   For ours - python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu $0/1/2/3$ --n 5 --net saved_models/class0.pth --ood_dataset cifar_non0_class  --indist_class 0 --proper_train_size 2000 --trials 5 --archi_type resnet18
### Optional - Train models on single class of CIFAR-10
    python train.py --cuda --outf saved_models --dataroot dataset --gpu $gpu_num$ --class_num $0-9$ --niter 200 --train_size 4500 --archi_type wrn
 
