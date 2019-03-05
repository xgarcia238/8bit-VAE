import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR,LambdaLR
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as functional

from data_utils import Compositions
from checkpoint import Checkpoint
import numpy as np

import time
import os
from math import exp

def Gaussian_KL_div(mu, var, var_prior, free_bits):
    """
    ----------------------------------------------------------------------------
    Computes the KL divergence D_{KL} (N(mu,var) || N(0,var_prior)).
    Args:
        mu (batch_size,latent_size)  : Mean for variational approximator.
        var (batch_size, latent_size): Variance for variational approximator.
        free_bits (float)            : Amount of nats we give to our KL.
    Returns:
        KL_div: Divergence between the N(mu, var) and N(0, var_prior)
        KL_reg: The KL divergence with the free bits reduction.
    ----------------------------------------------------------------------------
    """
    relu = nn.ReLU()
    KL_div = (torch.sum((var + mu**2)/var_prior + np.log(var_prior)-torch.log(var)-1)/2).detach()
    KL_reg = ((var + mu**2)/var_prior + np.log(var_prior) -torch.log(var)-1)/2
    KL_reg = relu(torch.sum(KL_reg,1).mean() - free_bits)
    return KL_div, KL_reg


def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def decay(x):
     return 0.01 + (0.99)*(0.9999)**x

class Trainer:

    """
    ----------------------------------------------------------------------------
    A trainer class which allows organized training of models with differnent
    parameters.
    ----------------------------------------------------------------------------
    """

    def __init__(self, exp_dir = 'experiment', score_type = 'exprsco',
                batch_size=64, random_seed=42, print_every=100, checkpoint_every=1000,
                samp_rate = 2000, KL_rate = 0.9999, free_bits = 60):

        #Prepare random seed.
        if random_seed is not None:
            torch.manual_seed(random_seed)

        #Prepare folder for experiment.
        if not os.path.isabs(exp_dir):
            exp_dir = os.path.join(os.getcwd(), exp_dir)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        self.exp_dir          = exp_dir
        self.optimizer        = None
        self.print_every      = print_every
        self.checkpoint_every = checkpoint_every
        self.loss_fn          = nn.CrossEntropyLoss()
        self.scheduler        = None
        self.batch_size       = batch_size
        self.samp_rate        = samp_rate
        self.KL_rate          = KL_rate
        self.free_bits        = free_bits
        self.vocab_size       = None

    def inverse_sigmoid(self,step):
        k = self.samp_rate
        if k == None:
            return 0
        if k == 1.0:
            return 1
        return k/(k + exp(step/k))

    def KL_annealing(self, step, start, end):
        return end + (start - end)*(self.KL_rate)**(step)

    def loss(self,step, encoder,decoder, batch, teacher_forcing):

        batch_size = batch.shape[0]
        mu, sig = encoder(batch)
        latent  = mu + sig*torch.randn_like(mu)
        logits  = decoder(latent, temp=None, x=batch, teacher_forcing = teacher_forcing, logits=True)
        KL_weight = self.KL_annealing(step, 0,0.2)
        KL_div, KL_reg = Gaussian_KL_div(mu,sig**2,1,self.free_bits)
        loss = KL_weight*KL_reg


        logit_loss = self.loss_fn(logits.view(-1,self.vocab_size), batch.view(-1))
        loss = loss + logit_loss


        correct = (torch.argmax(logits.view(-1,self.vocab_size), dim=1)==batch.view(-1)).float().sum()
        reconstruction_acc = correct/(batch_size*batch.shape[1])
        return loss,reconstruction_acc, KL_div/batch_size


    def train_batch(self,step, encoder, decoder, batch, teacher_forcing = True):
        loss, reconstruction_accuracy, KL_div = self.loss(step,encoder,decoder,batch, teacher_forcing)
        self.scheduler.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (loss.item(),reconstruction_accuracy*100, KL_div)

    def train_epochs(self, encoder, decoder, start_epoch,start_step, train_data,
    dev_data, end_epoch, log_file):

        #Prepare constants
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = {'batch_size': self.batch_size,
          'shuffle': True,
          'num_workers': 4,
          'drop_last': True}


        #Prepare dataloader and remaining constants.
        training_data   = DataLoader(train_data, **params)
        val_data        = DataLoader(dev_data, **params)
        steps_per_epoch = len(training_data)
        step            = start_step
        tot_steps       = steps_per_epoch*(end_epoch - start_epoch)
        elapsed_steps   = 0

        for epoch in range(start_epoch,end_epoch):
            print("Epoch: {:d}  Step: {:d}".format(epoch,step),file=open(log_file, 'a'))
            start = time.time()
            elapsed_steps = 0
            epoch_loss_total = 0.0
            reconstruction_accuracy_total = 0.0
            loss_total = 0.0
            KL_div_total = 0.0

            for batch in training_data:
                batch = batch.to(device)
                loss, reconstruction_accuracy, KL_div = self.train_batch(step, encoder, decoder, batch, self.inverse_sigmoid(step))
                loss_total += loss
                epoch_loss_total += loss
                reconstruction_accuracy_total += reconstruction_accuracy
                KL_div_total += KL_div
                step += 1
                elapsed_steps += 1

                if step%self.print_every == 0:
                    if elapsed_steps > self.print_every:
                        cnt = self.print_every
                    else:
                        cnt = elapsed_steps
                    loss_avg = loss_total /cnt
                    reconstruction_accuracy_avg = reconstruction_accuracy_total/cnt
                    KL_div_avg = KL_div_total/cnt
                    loss_total = 0
                    reconstruction_accuracy_total = 0
                    KL_div_total = 0
                    print(("Progress: {:.2f}%"
                    " Average Loss: {:2.2f}"
                    " Reconstruction Accuracy: {:2.2f}%"
                    " KL Divergence: {:2.2f}"
                    " ").format((elapsed_steps / steps_per_epoch)*100, loss_avg,reconstruction_accuracy_avg,KL_div_avg),file=open(log_file, 'a'))
                if step%self.checkpoint_every == 0:
                    print("Trying to checkpoint.")
                    Checkpoint( encoder   = encoder,
                                decoder   = decoder,
                                epoch     = epoch,
                                step      = step,
                                optimizer = self.optimizer,
                                scheduler = self.scheduler,
                                samp_rate = self.samp_rate,
                                KL_rate   = self.KL_rate,
                                free_bits = self.free_bits
                                ).save(self.exp_dir)
                    print("Checkpoint successful!")


            print("End of epoch. Time elapsed: " + timer(start, time.time()), file=open(log_file, 'a'))
            print("Average loss for this epoch: {:2.2f} ".format(epoch_loss_total/elapsed_steps), file=open(log_file, 'a'))
            Checkpoint( encoder   = encoder,
                        decoder   = decoder,
                        epoch     = epoch+1,
                        step      = step,
                        optimizer = self.optimizer,
                        scheduler = self.scheduler,
                        samp_rate = self.samp_rate,
                        KL_rate   = self.KL_rate,
                        free_bits = self.free_bits
                        ).save(self.exp_dir)

            #Now, compute validation.
            with torch.no_grad():
                reconstruction_accuracy_val = 0.0
                reconstruction_accuracy_val_nf = 0.0
                val_loss  = 0.0
                val_KL_tot = 0.0
                val_loss_nf = 0.0
                val_KL_tot_nf = 0.0
                count = 0
                for val_batch in val_data:
                    count += 1
                    val_batch = val_batch.to(device)
                    batch_loss, batch_accuracy, val_KL = self.loss(step, encoder,decoder, val_batch, 1)
                    batch_loss_nf, batch_accuracy_nf, val_KL_nf = self.loss(step, encoder, decoder, val_batch, 0)
                    val_loss += batch_loss
                    reconstruction_accuracy_val += batch_accuracy
                    val_KL_tot += val_KL

                    val_loss_nf += batch_loss_nf
                    reconstruction_accuracy_val_nf += batch_accuracy_nf
                    val_KL_tot_nf += val_KL_nf

                reconstruction_accuracy_val /= count
                val_loss /= count
                val_KL_tot /= count

                reconstruction_accuracy_val_nf /= count
                val_loss_nf /= count
                val_KL_tot_nf /= count
                print("Validation results: ", file=open(log_file, 'a'))
                print("Reconstruction Accuracy: {:2.2f}%"
                " Loss (Validation): {:2.2f}"
                " KL Divergence {:2.2f}".format(100*reconstruction_accuracy_val,val_loss,val_KL_tot), file=open(log_file, 'a'))

                print("Reconstruction Accuracy (NF): {:2.2f}%"
                " Loss (NF): {:2.2f}"
                " KL Divergence (NF) {:2.2f}".format(100*reconstruction_accuracy_val_nf,val_loss_nf,val_KL_tot_nf), file=open(log_file, 'a'))

    def train(self,encoder, decoder, n_epochs, train_data, dev_data,
                resume, optimizer, log_file):
        """
        ------------------------------------------------------------------------
        Args:
            encoder:                  Self explanatory.
            decoder:                  Self explanatory.
            n_epoch (int):            Number of epochs to train the model.
            train_data (Composition): Self explanatory.
            dev_data (Composition):   Self explanatory.
            resume (bool):            If true, load last checkpoint.
        ------------------------------------------------------------------------
        """
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.exp_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            encoder        = resume_checkpoint.encoder
            decoder        = resume_checkpoint.decoder
            start_epoch    = resume_checkpoint.epoch
            step           = resume_checkpoint.step
            self.scheduler = resume_checkpoint.scheduler
            self.optimizer = resume_checkpoint.optimizer
            self.samp_rate = resume_checkpoint.samp_rate
            self.KL_rate   = resume_checkpoint.KL_rate
            self.free_bits = resume_checkpoint.free_bits
            self.vocab_size = decoder.vocab_size
        else:
            self.optimizer = optimizer
            if optimizer is None:
                params = list(encoder.parameters()) + list(decoder.parameters())
                self.optimizer = Adam(params, lr=1e-3)
            self.scheduler = LambdaLR(self.optimizer,decay)
            self.vocab_size = decoder.vocab_size

            start_epoch = 1
            step = 0

        self.train_epochs(encoder, decoder, start_epoch, step, train_data, dev_data,
                        start_epoch + n_epochs, log_file)
        return encoder,decoder
