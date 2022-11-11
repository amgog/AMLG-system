

import pandas as pd
import numpy as np



def search_class(a):
    name = []
    for i in a:
        if i not in name:
            name.append(i)
    return name


def num_class(a, b):
    total_num = []
    for i in a:
        num = 0
        for j in b:
            if j == i:
                num += 1
        total_num.append(num)
    return total_num
def change_label(df,lab):
    df.insert(df.shape[1]-1,lab,df.pop(lab))
    # f = copy.deepcopy(df[lab])
    # df.drop(columns = lab)
    # ff = pd.concat([df,f],axis=1)
    return df

def divide_data(a, b, c):
    d = np.array(a.columns)
    # print(b)
    # print(c)
    # print(d)
    # print(d[-1])
    dividata = pd.DataFrame(columns=d)
    for i in range(len(b)):
        # for cla in a[d[-1]]:
        #     if cla == b[i]:
        #         print(cla.index())
        classdata = a[[cla == b[i] for cla in a[d[-1]]]]
        classdata = classdata.sample(n=c[i])
        dividata = pd.concat([dividata, classdata])
    return dividata

def read_sheet_name(path):
    f = pd.ExcelFile(path)
    f = f.sheet_names
    return f

def read_excel_file(path, name):
    f = pd.read_excel(path, sheet_name=name)
    f = pd.DataFrame(f)
    # title_name = np.array(f.columns)
    # title_number = len(title_name)
    # class_name = search_class(f[title_name[-1]])
    # class_num = len(class_name)
    # num_cla = num_class(class_name, f[title_name[-1]])
    # file_data = {
    #     'file_name': path,
    #     'sheet_name': name,
    #     'title_name': title_name,
    #     'title_number': title_number,
    #     'class_name': class_name,
    #     'class_num': class_num,
    #     'num_class': num_cla
    # }
    return f

def sum_excel_file(df):
    # f = pd.read_excel(path, sheet_name=name)
    f = df
    title_name = np.array(f.columns)
    sample_num = len(f.iloc[:,0])
    title_number = len(title_name)
    class_name = search_class(f[title_name[-1]])
    class_num = len(class_name)
    num_cla = num_class(class_name, f[title_name[-1]])
    file_data = {
        # 'file_name': path,
        # 'sheet_name': name,
        'sample_num': sample_num,
        'title_name': title_name,
        'title_number': title_number,
        'class_name': class_name,
        'class_num': class_num,
        'num_class': num_cla
    }
    return file_data


def collect_data(df, a, b, c):
    # f = pd.read_excel(path, sheet_name=name)
    f = df
    data = f[a]
    data2 = divide_data(data, b, c)
    return data2


def guiyihua(a, b):
    df1 = a[b[0:-1]].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    df1[b[-1]] = a[b[-1]]
    return df1

def biaozhunhua(a, b):
    df1 = a[b[0:-1]].apply(lambda x: (x - np.mean(x)) / np.std(x))
    df1[b[-1]] = a[b[-1]]
    return df1

def delenan(a,b):
    a = a.dropna(how='any',axis = b)
    return a

def percen_to_float(a):
    a = a.replace('%','')
    a = float(a)/100
    return a
# a = read_excel_file("D:\pycharm\project\pythonProject\四类数据477.xlsx", '四类数据')
#
# b = collect_data("D:\pycharm\project\pythonProject\四类数据477.xlsx", '四类数据',
#                  ['序号', '外套', '@2、衬衫', '@2、背心', '类别'], a['class_name'], [50, 50, 50, 50])
# print(b)
# b = guiyihua(b, ['序号', '外套', '@2、衬衫', '@2、背心', '类别'])
# print(b)
