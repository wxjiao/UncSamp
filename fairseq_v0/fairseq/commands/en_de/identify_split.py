# By xxx, 2020-May-22
#
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300



# Read token probability
print("\nReading in token probability ...")

def read_status(filename, pred_prob, pred_acc):
    with open(filename, 'r') as f:
        for line in tqdm(f):
            i, lp, acc = line.strip('\n').split('|||')
            i = int(i)
            lp_list_ = [float(lp_i) if lp_i not in ['nan'] else -100 for lp_i in lp.split()]
            lp_list = [lp_i_ for lp_i_ in lp_list_ if lp_i_ < 2.0]
            if len(lp_list) ==0:
                print(i, lp)
                continue
            if i in pred_prob.keys():
                pred_prob[i].append(lp_list)
            else:
                pred_prob[i] = [lp_list]

    return pred_prob, pred_acc

data = "/wmt14_en_de_base"
filename = "../../results" + data + "/sample_status/status_train_[BestStep].txt"
pred_prob, pred_acc = dict(), dict()
if os.path.isfile(filename):
    print(filename)
    pred_prob, pred_acc = read_status(filename, pred_prob, pred_acc)
else:
    print("No file found! Pleae check the path!")
    exit()



# Compute sentence probability
print("\nComputing sentence probability ...")

ids = []
p_sent_mean = []
p_sent_std = []

dict_p_tok_mean = dict()
dict_p_tok_std = dict()

num_ckp = (len(pred_prob[0])-1)
print("Number of checkpoint change {}".format(num_ckp))

max_id =  max(pred_prob.keys())
print("Recorded examples {}, max_id {}".format(len(pred_prob), max_id))

for i in range(max_id + 1):  # len(pred_prob) <= len(data), some examples may be skipped.
    if i % 500000 == 0:
        print("Step {}".format(i))
    if i in pred_prob.keys():
        ids.append(i)
        # forget
        lp_list = pred_prob[i]
        lp_list = np.array(lp_list) # num_ckp x L
        lp_sent_i = np.mean(lp_list, axis=1)   # num_ckp x 1
        p_sent_i = np.exp(lp_sent_i)
        p_sent_mean.append(p_sent_i.mean())
print("Done reading!")
print(len(p_sent_mean), len(p_sent_std))

test_idx = 10
data_id = ids[test_idx]
print("A test example: idx {}; p {}".format(test_idx, p_sent_mean[test_idx]))



# Sort by sentence probability
print("\nSorting sentence pairs by sentence probability ...")

p_sent_mean = np.array(p_sent_mean)
ids = np.array(ids)

sorted_idx = np.argsort(p_sent_mean)
sorted_ids = ids[sorted_idx]
sorted_p_sent_mean = p_sent_mean[sorted_idx]



# Record order of examples
print("\nWriting the order of sentence pairs ...")

sample_order_path = "../../results" + data + "/sample_status"  + "/sample_order_prob_[BestStep].txt"
with open(sample_order_path, 'w') as file:
    for od in sorted_ids:
        file.write(str(od)+'\n')
        
        
        
# Plot the sorted sentence probability
print("\nPlotting and saving the sorted sentence ...")

fig3 = plt.figure(figsize=(5,3))
ax31 = fig3.add_subplot(111)
ax31.plot(np.linspace(0,100,len(sorted_p_sent_mean)), sorted_p_sent_mean, color='b')
ax31.set_xlabel('Training Examples (%)')
ax31.set_xticks(range(0,110,10))
ax31.set_yticks([0.0, 0.5, 1.0])
ax31.set_ylabel('Prediction Confidence')
fig_path = "../../results" + data + "/sample_status" + "/sentence-prob.png"
fig3.savefig(fig_path, dpi=300)



# Read training data
print("\nReading in training data ...")

def read_train(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines

src_raw_path = "../../dataset" + data + '/train.en'
tgt_raw_path = "../../dataset" + data + '/train.de'
src_raw = read_train(src_raw_path)
tgt_raw = read_train(tgt_raw_path)
print(len(src_raw), len(tgt_raw))



# Split and save inactive and active examples
print("\nSpliting inactive examples and active examples ...")

def write_train(data, selected, path):
    with open(path, 'w') as file:
        for i in tqdm(selected):
            file.write(data[i])

rate = 0.1
num_rm = int(len(sorted_ids)*rate)
print("Number of inactive examples: ", num_rm)

inactive_ids = sorted_ids[:num_rm]
active_ids = sorted_ids[num_rm:]
print(len(inactive_ids), inactive_ids[:10])
print(len(active_ids), active_ids[:10])

split_data = "/wmt14_en_de_base_identified"
split_path = "../../dataset" + split_data
if not os.path.isdir(split_path):
      os.mkdir(split_path)

path_s_in = split_path + '/inactive.en'
path_t_in = split_path + '/inactive.de'
write_train(src_raw, inactive_ids, path_s_in)
write_train(tgt_raw, inactive_ids, path_t_in)

path_s_ac = split_path + '/active.en'
path_t_ac = split_path + '/active.de'
write_train(src_raw, active_ids, path_s_ac)
write_train(tgt_raw, active_ids, path_t_ac)

print("All done!")



