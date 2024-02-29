import os

dir_path = 'caltech256/256_ObjectCategories'

label = []

id = 0
for dir in os.listdir(dir_path):
    l = dir.split('.')[-1]
    label.append(f'{id}:{l}\n')
    id+=1

with open('caltech256_label.txt','w') as f:
    f.writelines(label)

def read_label():
    cls_dict = {}
    with open('caltech256_label.txt','r') as f:
        lines = f.readlines()
        for l in lines:
            cls_dict[l.split(':')[0]] = l.split(':')[1].strip()
    return  cls_dict