import os,sys
sys.path.append(os.path.dirname(__file__))
print(sys.path)

src = []
fp32 = []
int2 = []

def get_acc(file_name, target_list):
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        content = line.split()
        target_list.append(float(content[2]))

get_acc('result_fp.txt',src)
get_acc('result_fp32.txt', fp32)
get_acc('result_int2.txt', int2)

import matplotlib.pyplot as plt

epoch = range(200)
plt.plot(epoch,src,c='red',label='fp32')
plt.plot(epoch,fp32,c='blue',label='fp32_quant')
plt.plot(epoch,int2,c='yellow',label='int2_quant')
plt.legend(loc='best')

plt.xlabel("epoch", fontdict={'size': 16})
plt.ylabel("acc", fontdict={'size': 16})
plt.show()
plt.savefig('acc.png')
