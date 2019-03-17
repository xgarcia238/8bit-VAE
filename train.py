import torch
import VAE_Trainer
#from multitrackVAE import Encoder, Decoder
from MusicVAE_TF import Encoder, Decoder
from data_utils import CompactCompositions
from checkpoint import Checkpoint

#Prepare relevant constants.
exp_dir = "experiment"
log_file = "experiment_log"
n_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Prepare data.
score_type = "seprsco"
train_data_path = '50_12_seprsco_train_no_pad_TR'
dev_data_path = '50_12_seprsco_val_no_pad_TR'
train_comp = CompactCompositions(train_data_path)
val_comp = CompactCompositions(dev_data_path)

#Prepare trainer.
Ash_Ketchum = VAE_Trainer.Trainer(exp_dir = exp_dir, score_type = score_type,
            batch_size=128, random_seed=42, print_every=100, checkpoint_every=10000,
            samp_rate = None, KL_rate = 0.99991, free_bits = 125)

#Check to see if we've already started the experiment
resume = False
if resume:
    latest_checkpoint_path = Checkpoint.get_latest_checkpoint("experiment_1")
    print("Resuming training...")
    Ash_Ketchum.train(None,None,n_epochs,train_comp,val_comp,True,None,log_file)
else:
    vocab_size = 275
    encoder_hidden_size = 256
    decoder_hidden_size = 512
    latent_size = 128
    seq_size = 52
    num_layers = 2
    encoder = Encoder(vocab_size, encoder_hidden_size, latent_size,
                        seq_size, num_layers).to(device)
    decoder = Decoder(latent_size, decoder_hidden_size, vocab_size,
                        num_layers, seq_size).to(device)

    Ash_Ketchum.train(encoder,decoder,n_epochs, train_comp, val_comp, resume=False,
                        optimizer=None, log_file =log_file)
