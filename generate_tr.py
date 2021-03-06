import os
import torch
from interpolation import interpolation

path = '2019_03_02_19_39_56'
#path = '2019_03_13_17_16_31'
temp = None
seconds = 3
name = "test_sample"
score_type = "seprsco"
data_path = '52_12_seprsco_train_no_pad_TR'

#n2 = 4173
n1= None
#n2 = 87964
n2 = 108961
#generate_nes_score(path, seconds, name, temp, score_type)

"""
Good songs:

(4173, 2953)
(66345,25100,10)
(210000,21500,7)

###
52:

(121820, 108961,10)
(158027, 4668,10)
(108961, 2279195. 15)


"""
with torch.no_grad():
    interpolation(data_path, path, temp, seconds, name,
                song_id1 = n1, song_id2 = n2, n_steps = 10, TR = True)
