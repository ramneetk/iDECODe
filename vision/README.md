# OOD and adversarial detection on vision dataset

  This code is build on top of AVT model code from qi et al. 2019's code for AVT model, https://github.com/maple-research-lab/AVT-pytorch

### This code requires Python==3.8. To install the dependencies please run:
    conda create -n vision python=3.8
    conda activate vision
    pip install -r requirements.txt
    
## SOTA results (Table 2)

    cd SOTA 
    Follow instructions in SOTA/README.md
    
## Ablation and adversarial results (Tables 1, 5 and Figures 1, 2, 3, 4)    
    
    cd ablation_and_adversarial  
    Follow instructions in ablation_and_adversarial/README.md
   
## Optional: Train the ResNet-34 model on CIFAR-10 dataset
    
   python train.py --cuda --outf $output dir to save the model$ --gpu $gpu no.$ --proper_train_size 45000
