from checkpoint import Checkpoint
from data_utils import CompactCompositions
from preprocessing import postprocessing
import torch
import numpy as np
import pickle

def unit_vector(vector):
    return vector / torch.sum(vector**2)**0.5

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return torch.acos(torch.dot(v1_u[0], v2_u[0])).unsqueeze(0)

def spherical_interpolation(code1,code2,t):
    angle = angle_between(code1,code2)
    return (torch.sin((1-t)*angle)*code1 + torch.sin(t*angle)*code2)/torch.sin(angle)

#def spherical_interpolation(code1,code2,t):
#    return (1-t)*code1 + t*code2

def interpolation(data_path, checkpoint_path, temp, seconds, name,
                song_id1 = None, song_id2 = None, n_steps = 20):
    #Load decoder.
    cp = Checkpoint.load(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = cp.encoder.to(device).eval()
    decoder = cp.decoder.to(device).eval()
    rate = 24.0
    nsamps = 44100*seconds

    #Load data and create points. If no path, make new ones.
    comps = CompactCompositions(data_path)
    n_comps = len(comps)
    if song_id1 is None:
        song_id1 = np.random.randint(0,n_comps)
    if song_id2 is None:
        song_id2 = np.random.randint(0,n_comps)

    n_comps  = len(comps)
    begin    = comps[song_id1]

    end      = comps[song_id2]


    begin_score = postprocessing([begin.numpy()],32)
    end_score = postprocessing([end.numpy()],32)
    #Store for comparison.


    b_nsamps = 44100*begin_score.shape[0]//24
    e_nsamps = 44100*end_score.shape[0]//24
    with open('begin' + '.pickle', 'wb') as f1:
        pickle.dump((rate,b_nsamps,begin_score), f1, protocol=2)
    with open('end' + '.pickle', 'wb') as f2:
        pickle.dump((rate,e_nsamps,end_score), f2, protocol=2)


    #with open('test_begin' + '.pickle', 'wb') as f3:
    ##    pickle.dump(decoder(lat_1))

    #Make code.

    A = torch.cat([begin.unsqueeze(0),begin.unsqueeze(0)])
    B = torch.cat([end.unsqueeze(0),end.unsqueeze(0)],0)
    lat_1,_ = encoder(A)
    recon_1 = decoder(lat_1, temp=None, x=A, teacher_forcing=True)
    #recon_1 = torch.argmax(recon_1, 2)
    lat_2, _ = encoder(B)
    recon_2 = decoder(lat_2, temp=None, x=B, teacher_forcing=True)
    #recon_2 = torch.argmax(recon_2,2)

    steps = [spherical_interpolation(lat_1,lat_2,float(t)/float(n_steps)) for t in range(1,n_steps)]
    steps = [decoder(step, temp, x=None, teacher_forcing=False) for step in steps]
    steps = [step[0].unsqueeze(0) for step in steps]
    steps = torch.cat(steps)
    steps = torch.cat((recon_1[0].unsqueeze(0), steps, recon_2[0].unsqueeze(0)),0)
    steps = postprocessing(steps,32)
    time = 44100*steps.shape[0]//24
    print(steps.shape)

    with open(name + '.pickle', 'wb') as f:
        pickle.dump((rate,time,steps), f, protocol=2)
