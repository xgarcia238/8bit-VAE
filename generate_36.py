import os
import torch
from interpolation import interpolation

#path = '2019_02_20_14_44_55'
#path = '2019_02_20_17_50_18'
path = '2019_02_20_21_08_40'
temp = None
seconds = 3
name = "test_sample"
score_type = "seprsco"
data_path = '36_12_seprsco_train_no_pad'

#n2 = 4173
n1= 9690
#n1 = 4173
#n2 = 2953
#n2 = 2953
#n2 = 33614#
#n1 =  2953
#n1 = 0
#n2 = 210000
n2 = 11778
#generate_nes_score(path, seconds, name, temp, score_type)

"""
Good songs:

(4173, 2953)
(66345,25100,10)
(210000,21500,7)
"""
with torch.no_grad():
    interpolation(data_path, path, temp, seconds, name,
                song_id1 = n1, song_id2 = n2, n_steps = 10)
