**VAD-VAE**
This repo contains the PyTorch code for IEEE TAC accepted paper: "Disentangled Variational Autoencoder for Emotion Recognition in Conversations".

**Preparation**
1. Set up the Python 3.7 environment

2. Build the dependencies with the following code:
pip install -r requirements.txt

3. Download the training data from [this link](https://drive.google.com/file/d/1HqgroEAvfZcGplBbtxOzhko2-iyYbr9s/view?usp=sharing) and put it under the VAD-VAE dir.

**Training and evaluation**

Train on IEMOCAP:
python main.py --DATASET IEMOCAP --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4 --kl_weight 0.001 --bart_model_checkpoint facebook/bart-large --model_save_dir ./model_save_dir/IEMOCAP --mode train --SEED 42 --CUDA --mi_loss

Then evaluate on IEMOCAP:
python main.py --DATASET IEMOCAP --model_checkpoint roberta-large --alpha 0.8 --BATCH_SIZE 4 --kl_weight 0.001 --bart_model_checkpoint facebook/bart-large --mode eval --model_load_path ./model_save_dir/IEMOCAP/model_state_dict_2.pth --SEED 42 --CUDA --mi_loss

Train on MELD:
python main.py --DATASET MELD --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 4 --kl_weight 0.001 --bart_model_checkpoint facebook/bart-large --model_save_dir ./model_save_dir/MELD --mode train --SEED 2 --CUDA

Then evaluate on MELD:
python main.py --DATASET MELD --model_checkpoint roberta-large --alpha 0.8 --BATCH_SIZE 4 --kl_weight 0.001 --bart_model_checkpoint facebook/bart-large --mode eval --model_load_path ./model_save_dir/MELD/model_state_dict_4.pth --SEED 42 --CUDA

Train on DailyDialog:
python main.py --DATASET DailyDialog --model_checkpoint roberta-large --alpha 0.8 --NUM_TRAIN_EPOCHS 5 --BATCH_SIZE 16 --kl_weight 0.001 --bart_model_checkpoint facebook/bart-large --model_save_dir ./model_save_dir/DailyDialog --mode train --SEED 42 --CUDA --mi_loss

Then evaluate on DailyDialog:
python main.py --DATASET DailyDialog --model_checkpoint roberta-large --alpha 0.8 --BATCH_SIZE 16 --kl_weight 0.001 --bart_model_checkpoint facebook/bart-large --mode eval --model_load_path ./model_save_dir/DailyDialog/model_state_dict_3.pth --SEED 42 --CUDA --mi_loss