# AUDIO OOD detection
  This code is build on top of T. Iqbal et. al's "Learning with OOD Audio"- https://github.com/tqbl/ood_audio

### This code requires Python >=3.6. To install the dependencies, run:
    conda create -n audio python=3.6
    conda activate audio
    pip install -r requirements.txt
  
### Download FSDnoisy18k dataset:
    sh scripts/download_dataset.sh
  
### Extract features of the training and test set
    sh scripts/extract.sh

### Trained models
 ## Link for downloading the trained audio models - https://drive.google.com/file/d/1z7PwOp8GnHUNURLHVJpgN5CkPDZQzfdb/view?usp=drive_link
    mkdir ./_workspace/models
    mv audio_models.zip ./_workspace/models
    unzip ./_workspace/models/audio_models.zip -d  ./_workspace/models

### Generate results for Table 4:
  # For ICAD
    python ood_audio/check_ood_non_conf.py --training_id class_set_0 --class_set_name 0 --n 1 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_1 --class_set_name 1 --n 1 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_2 --class_set_name 2 --n 1 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_3 --class_set_name 3 --n 1 --trials 5
   # For Ours with |V(x)| = 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_0 --class_set_name 0 --n 5 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_1 --class_set_name 1 --n 5 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_2 --class_set_name 2 --n 5 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_3 --class_set_name 3 --n 5 --trials 5
   # For SBP, Base Score and Ours with |V(x)| = 20
    python ood_audio/check_ood_non_conf.py --training_id class_set_0 --class_set_name 0 --n 20 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_1 --class_set_name 1 --n 20 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_2 --class_set_name 2 --n 20 --trials 5
    python ood_audio/check_ood_non_conf.py --training_id class_set_3 --class_set_name 3 --n 20 --trials 5

### Optional: Training audio models on the four sets
    python ood_audio/main.py train --mask manually_verified=1 --training_id class_set_$0/1/2/3$ --augment True --relabel False --model vgg --class_set_name $0/1/2/3$ --n_epochs 30 --overwrite True

