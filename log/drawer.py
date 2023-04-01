import os,sys
sys.path.append(os.path.dirname(__file__))
print(sys.path)

src = []
int8 = []
int2 = []
src_int = []

def get_acc(file_name, target_list):
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        content = line.split()
        target_list.append(float(content[2]))

get_acc('result_fp.txt',src)
get_acc('result_int8_bits[2,3,4,8]', int8)
get_acc('result_int2_bits[2,3,4,8].txt', int2)
get_acc('result_int.txt', src_int)
import matplotlib.pyplot as plt

epoch = range(200)
plt.plot(epoch,src,c='red',label='fp32')
plt.plot(epoch,int8,c='pink',label='int8_quant')
plt.plot(epoch,int2,c='cyan',label='int2_quant')
plt.plot(epoch,src_int,c='blue',label='int2')
plt.legend(loc='best')

plt.xlabel("epoch", fontdict={'size': 16})
plt.ylabel("acc", fontdict={'size': 16})
plt.show()
plt.savefig('acc.png')
