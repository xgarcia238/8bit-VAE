from preprocessing import dataset_builder
import time

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

num_steps = 52
window_size = 24
note_size = 77
threshold = 32
TR = True
sparse = True
#First, prepare training data.
print("Preparing validation data.")
start = time.time()
input_folder = 'nesmdb24_seprsco/valid'
score_type = 'seprsco'
label = 'val_sparse'
output_folder = str(num_steps) + "_" + str(window_size) + "_" + score_type + "_" + label

size = dataset_builder(input_folder, output_folder, score_type,threshold, note_size,
                    num_steps, window_size, min_length = 3,TR=TR)


#Next, prepare validation data.
print("Done.")
print(timer(start, time.time()))

print("Preparing training data.")
start = time.time()
input_folder = 'nesmdb24_seprsco/train'
score_type = 'seprsco'
label = 'train_sparse'
output_folder = str(num_steps) + "_" + str(window_size) + "_" + score_type + "_" + label
size = dataset_builder(input_folder, output_folder, score_type,threshold, note_size,
                            num_steps, window_size, min_length = 3,TR=TR)


print("Done")
print(timer(start, time.time()))
