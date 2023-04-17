import sys,os
sys.path.append(os.path.dirname(__file__))

from scipy.stats.stats import kendalltau
import matplotlib.pyplot as plt

def get_sample_acc(filename):
    file = open(filename)
    lines = file.readlines()
    scratch_acc = 0
    for line in lines:
        contents = line.split()
        if contents[0] == '0':
            sample_acc = float(contents[2])
    return sample_acc

def get_scratch_acc(filename):
    file = open(filename)
    lines = file.readlines()
    scratch_acc = 0
    for line in lines:
        contents = line.split()
        if int(contents[0]) == 140 :
            scratch_acc = float(contents[2]) 
    return scratch_acc


if __name__ == '__main__':
    sample_accs = []
    scratch_accs = []
    for i in range(8):
        sample_filename = f'sample{i}.txt'
        scratch_filename = f'sample{i}_scratch.txt'
        sample_acc = get_sample_acc(sample_filename)
        scratch_acc = get_scratch_acc(scratch_filename)
        sample_accs.append(sample_acc)
        scratch_accs.append(scratch_acc)


    
    fig,ax = plt.subplots()
    ax.scatter(sample_accs,scratch_accs)
    ax.set_xlabel('Sampled network')
    ax.set_ylabel('inividual network')
    fig.savefig('kendal.png')

    print(kendalltau(sample_accs,scratch_accs))
