import os,sys
sys.path.append(os.path.dirname(__file__))
print(sys.path)

src = []
int8 = []
int2 = []
int8_quant = []
int2_quant = []
int8_para = []
int2_para = []
src_int = []

def get_acc(file_name, target_list):
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        if not line[0]=='#':
            content = line.split()
            epoch = int(content[0])
            if epoch % 5 ==0:
                target_list.append(float(content[2]))

get_acc('result_fp.txt',src)

get_acc('result_int8_more8loss.txt', int8_quant)
get_acc('result_int2_more8loss.txt', int2_quant)
get_acc('result_int8_parallel.txt', int8_para)
get_acc('result_int2_parallel.txt', int2_para)

import matplotlib.pyplot as plt

epoch = [i for i in range(1,201,5)]

plt.plot(epoch,int8_quant,c='blue',label='int8_single')
plt.plot(epoch,int2_quant,c='blue',label='int2_single')
plt.plot(epoch,int8_para,c='pink',label='int8_para')
plt.plot(epoch,int2_para,c='pink',label='int2_para')

plt.legend(loc='best')

plt.xlabel("epoch", fontdict={'size': 16})
plt.ylabel("acc", fontdict={'size': 16})
plt.show()
plt.savefig('single&para.png')
