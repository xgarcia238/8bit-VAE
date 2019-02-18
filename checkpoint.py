import torch
import time
import os
import shutil

class Checkpoint:
    """
    ----------------------------------------------------------------------------
    A Checkpoint class.

    The purpose is to be organized with our experiments, as well to save and
    load progress in training.
    ----------------------------------------------------------------------------
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    ENCODER_NAME = 'encoder.pt'
    DECODER_NAME = 'decoder.pt'

    def __init__(self, encoder, decoder, epoch, step, optimizer, scheduler,
                samp_rate, KL_rate, free_bits, path=None):
        self.encoder   = encoder
        self.decoder   = decoder
        self.epoch     = epoch
        self.step      = step
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.samp_rate = samp_rate
        self.KL_rate   = KL_rate
        self.free_bits = free_bits
        self._path     = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("This checkpoint has not been saved.")
        return self._path

    def save(self, experiment_dir):

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        #If path exists, erase the whole thing nad make a new one.
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler,
                    'samp_rate': self.samp_rate,
                    'KL_rate': self.KL_rate,
                    'free_bits': self.free_bits
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.encoder, os.path.join(path, self.ENCODER_NAME))
        torch.save(self.decoder, os.path.join(path, self.DECODER_NAME))

        return path

    @classmethod
    def load(cls, path):

        #Check if GPU is available.
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            encoder = torch.load(os.path.join(path, cls.ENCODER_NAME))
            decoder = torch.load(os.path.join(path, cls.DECODER_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),map_location=lambda storage, loc: storage)
            encoder = torch.load(os.path.join(path, cls.ENCODER_NAME),map_location=lambda storage, loc: storage)
            decoder = torch.load(os.path.join(path, cls.DECODER_NAME),map_location=lambda storage, loc: storage)

        #Make RNN parameters contiguous.
        encoder.flatten_parameters()
        decoder.flatten_parameters()
        return Checkpoint(encoder= encoder,
                        decoder=decoder,
                        epoch= resume_checkpoint['epoch'],
                        step=resume_checkpoint['step'],
                        optimizer= resume_checkpoint['optimizer'],
                        scheduler = resume_checkpoint['scheduler'],
                        samp_rate = resume_checkpoint['samp_rate'],
                        KL_rate = resume_checkpoint['KL_rate'],
                        free_bits = resume_checkpoint['free_bits'],
                        path= path)
    @classmethod
    def get_latest_checkpoint(cls, exp_path):
        checkpoints_path = os.path.join(exp_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
