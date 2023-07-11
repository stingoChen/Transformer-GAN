import numpy as np


def read_data(file_path, class_):
    f1 = open(file_path)
    data = []
    lines = f1.readlines()
    for line in lines:
        tmp = []
        line1 = line.strip().split(", ")
        for n, i in enumerate(line1):
            if n == 0:
                tmp.append(float(i[1:]))
            elif n == 10:
                tmp.append(float(i[:-1]))
            else:
                tmp.append(float(i))
        tmp += [class_]
        data.append(tmp)
    data = np.array(data)
    f1.close()
    return data


data1 = read_data("./f1_40.txt", 1)
data2 = read_data("./f2_40.txt", 2)
data3 = read_data("./f3_40.txt", 3)
data5 = read_data("./f5_40.txt", 4)

# data4 = read_data("./30/f4_30.txt", 4)
# data5 = read_data("./f5_40.txt", 5)

data_s = np.concatenate((data1, data2), axis=0)
data_s = np.concatenate((data_s, data3), axis=0)
data_s = np.concatenate((data_s, data5), axis=0)
print(data_s)
# data_s = np.concatenate((data_s, data4), axis=0)
# data_s = np.concatenate((data_s, data5), axis=0)

