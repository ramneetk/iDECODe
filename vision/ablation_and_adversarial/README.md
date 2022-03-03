### Download Imagenet and LSUN datasets, "Imagenet_resize.tar.gz" and "LSUN_resize.tar.gz"
## LSUN_resize - https://drive.google.com/file/d/1vD36F8JA0PN6cUGfxrBJmDZ-dKG7F-2Q/view?usp=sharing
## Imagenet_resize - https://drive.google.com/file/d/1ZSDUTV2z_nSinXampJKsCtwJPSI1IHjM/view?usp=sharing
    mkdir data 
    mv Imagenet_resize.tar.gz ./data
    tar -xf data/Imagenet_resize.tar.gz -C ./data
    mv LSUN_resize.tar.gz ./data
    tar -xf data/LSUN_resize.tar.gz -C ./data

### Download trained model on CIFAR-10 dataset, "cifar10.pth" - https://drive.google.com/file/d/1X6CtJO4RyArTFhC2yHoThU42bTgM47Ud/view?usp=sharing
    mv cifar10.pth .

### Adversarial samples: FGSM, BIM, DeepFool, CW on ResNet and DenseNet architectures
#### This samples are generated from https://github.com/pokaxpoka/deep_Mahalanobis_detector
## Download adversarial data - https://drive.google.com/file/d/1aVohIIDejp6xAdcyekinlM0LzU2PKFbY/view?usp=sharing
    mv adversarial_data.zip .
    unzip adversarial_data.zip

### Generating TNR and AUROC results in Table 1 for ablation studies on CIFAR10 

    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset IMAGENET --proper_train_size 45000 --trials 5
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset CIFAR100 --proper_train_size 45000 --trials 5
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset LSUN --proper_train_size 45000 --trials 5
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset SVHN --proper_train_size 45000 --trials 5
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset Places365 --proper_train_size 45000 --trials 5
    

### Generating TNR/AUROC VS n plots for CIFAR10 dataset, the graphs are saved as $imagenet/cifar100/lsun/svhn/places_tnr/roc.pdf$ (Figure 2, 3)
    python check_performance_n.py --cuda --gpu 0 --net ./cifar10.pth --n 20 --ood_dataset IMAGENET
    python plot_imagenet_roc.py
    python plot_imagenet_tnr.py
    python check_performance_n.py --cuda --gpu 0 --net ./cifar10.pth --n 20 --ood_dataset CIFAR100
    python plot_cifar100_roc.py
    python plot_cifar100_tnr.py
    python check_performance_n.py --cuda --gpu 0 --net ./cifar10.pth --n 20 --ood_dataset LSUN
    python plot_lsun_roc.py
    python plot_lsun_tnr.py
    python check_performance_n.py --cuda --gpu 0 --net ./cifar10.pth --n 20 --ood_dataset SVHN
    python plot_svhn_roc.py
    python plot_svhn_tnr.py
    python check_performance_n.py --cuda --gpu 0 --net ./cifar10.pth --n 20 --ood_dataset Places365 
    python plot_places365_roc.py
    python plot_places365_tnr.py

### Generating box plots, the graph is saved as 1000.pdf and 2000.pdf (Figure 1, 4)
    python fnr_at_fixed_epsilon.py --net ./cifar10.pth --cuda --gpu 0  --n 5 --trials 5 --cal_set_size_trial 1000 --file_name 1000.pdf
    python fnr_at_fixed_epsilon.py --net ./cifar10.pth --cuda --gpu 0  --n 5 --trials 5 --cal_set_size_trial 2000 --file_name 2000.pdf

### Generating results on adversarial samples, (Table 5)
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/ResNet34/adv_data_ResNet34_cifar10_FGSM.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/ResNet34/adv_data_ResNet34_cifar10_BIM.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/ResNet34/adv_data_ResNet34_cifar10_DeepFool.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/ResNet34/adv_data_ResNet34_cifar10_CWL2.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/DenseNet/adv_data_DenseNet3_cifar10_FGSM.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/DenseNet/adv_data_DenseNet3_cifar10_BIM.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/DenseNet/adv_data_DenseNet3_cifar10_DeepFool.pth
    python check_OOD.py --cuda --gpu 0 --net ./cifar10.pth --n 5 --ood_dataset adv_cifar10 --proper_train_size 45000 --adv_data_root adversarial_data/DenseNet/adv_data_DenseNet3_cifar10_CWL2.pth

    



