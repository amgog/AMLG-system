from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

from sklearn import svm

tt = {
    'svm': ['kernal', 'decision_function_shape', 'degree', 'coef0', 'gamma', 'C', 'tol', '参数8', '参数9'],
    'dct': ['criterion', 'splitter', 'max_leaf_nodes', 'max_depth', 'min_impurity_decrease', 'min_samples_split',
            'min_samples_leaf', 'random_state', '参数9'],
    'ada': ['algorithm','base_estimator','n_estimators','learning_rate','random_state','参数6','参数7','参数8','参数9'],
    'ran': ['criterion','bootstrap','max_depth','min_samples_leaf','min_samples_split','max_features',
            'max_leaf_nodes','n_estimators','参数9']
}

def coef_format(m, coef):
    if m == 'svm':
        if coef[4] != 'auto':
            coef[4] = float(coef[4])
        return coef
    if m == 'ran':
        if coef[2] != 'None':
            coef[2] = int(coef[2])
        else:
            coef[2] = None
        if coef[5] != 'None':
            coef[5] = int(coef[5])
        else:
            coef[5] = None
        if coef[6] != 'None':
            coef[6] = int(coef[6])
        else:
            coef[6] = None
        return coef
    if m == 'ada':
        if coef[1] == 'None':
            coef[1] = None
        if coef[4] != 'None':
            coef[4] = int(coef[4])
        else:
            coef[4] = None
        return coef
    if m == 'dct':
        if coef[2] != 'None':
            coef[2] = int(coef[2])
        else:
            coef[2] = None
        if coef[3] != 'None':
            coef[3] = int(coef[3])
        else:
            coef[3] = None
        if coef[7] != 'None':
            coef[7] = int(coef[7])
        else:
            coef[7] = None
        return coef


def get_classweight(a):
    c = list(map(float, a))
    k = c[0]
    cw = list(map(lambda x: x/k, c))
    return cw


def svm_classify(data, coef, r,dic,d):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    y = y.astype(str).map(dic)
    r = list(map(int, r))
    if r[2] != 0:
        X, X_verify, Y, Y_verify = train_test_split(
            x, y, test_size=r[2] / sum(r), random_state=1, shuffle=True, stratify=y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=Y)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=y)

    # clf = svm.SVC(C= 1.5,gamma='auto', kernel='poly', coef0=1, degree=2, decision_function_shape='ovo')
    clf = svm.SVC(kernel=coef[0],decision_function_shape=coef[1],degree=int(coef[2]),coef0 = float(coef[3]),gamma=coef[4],
                  C=float(coef[5]),tol=float(coef[6]),class_weight=d)
    clf.fit(X_train, Y_train)
    x_pre = clf.predict(x)
    X_train_pre = clf.predict(X_train)
    X_test_pre = clf.predict(X_test)
    if r[2] != 0:
        X_verify_pre = clf.predict(X_verify)
    preci_all = precision_score(x_pre, y, average='weighted')
    preci_cla = precision_score(x_pre, y, average=None)
    preci_train_all = precision_score(X_train_pre, Y_train, average='weighted')
    preci_train_cla = precision_score(X_train_pre, Y_train, average=None)
    preci_test_all = precision_score(X_test_pre, Y_test, average='weighted')
    preci_test_cla = precision_score(X_test_pre, Y_test, average=None)

    recall_all = recall_score(x_pre, y, average='weighted')
    recall_cla = recall_score(x_pre, y, average=None)
    recall_train_all = recall_score(X_train_pre, Y_train, average='weighted')
    recall_train_cla = recall_score(X_train_pre, Y_train, average=None)
    recall_test_all = recall_score(X_test_pre, Y_test, average='weighted')
    recall_test_cla = recall_score(X_test_pre, Y_test, average=None)
    if r[2] != 0:
        preci_verify_all = precision_score(X_verify_pre, Y_verify, average='weighted')
        preci_verify_cla = precision_score(X_verify_pre, Y_verify, average=None)
        recall_verify_all = recall_score(X_verify_pre, Y_verify, average='weighted')
        recall_verify_cla = recall_score(X_verify_pre, Y_verify, average=None)
    else:
        preci_verify_all = 0.00
        preci_verify_cla = 0.00
        recall_verify_all = 0.00
        recall_verify_cla = 0.00
    res = {
        '总体精确率': preci_all,
        '分类总体精确率': preci_cla,
        '训练精确率': preci_train_all,
        '分类训练精确率': preci_train_cla,
        '测试精确率': preci_test_all,
        '分类测试精确率': preci_test_cla,
        '验证精确率': preci_verify_all,
        '分类验证精确率': preci_verify_cla,
        '总体召回率': recall_all,
        '分类总体召回率': recall_cla,
        '训练召回率': recall_train_all,
        '分类训练召回率': recall_train_cla,
        '测试召回率': recall_test_all,
        '分类测试召回率': recall_test_cla,
        '验证召回率': recall_verify_all,
        '分类验证召回率': recall_verify_cla
    }
    return res


def random_forest(data, coef, r,dic,d):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    y = y.astype(str).map(dic)
    r = list(map(int, r))
    if r[2] != 0:
        X, X_verify, Y, Y_verify = train_test_split(
            x, y, test_size=r[2] / sum(r), random_state=1, shuffle=True, stratify=y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=Y)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=y)

    # clf = RandomForestClassifier(n_estimators=1,
    #                             criterion='friedman_mse',
    #                             random_state=1,
    #                             n_jobs=-1)
    # 'ran': ['criterion','bootstrap','max_depth','min_samples_leaf','min_samples_split','max_features',
    #         'max_leaf_nodes','n_estimators','参数9']
    clf = RandomForestClassifier(criterion=coef[0],bootstrap=coef[1],max_depth=coef[2],min_samples_leaf = int(coef[3]),
                                min_samples_split=int(coef[4]),max_features=coef[5],max_leaf_nodes=coef[6],
                                 n_estimators = int(coef[7]),class_weight=d)
    clf.fit(X_train, Y_train)
    x_pre = clf.predict(x)
    X_train_pre = clf.predict(X_train)
    X_test_pre = clf.predict(X_test)

    if r[2] != 0:
        X_verify_pre = clf.predict(X_verify)
    preci_all = precision_score(x_pre, y, average='weighted')
    preci_cla = precision_score(x_pre, y, average=None)
    preci_train_all = precision_score(X_train_pre, Y_train, average='weighted')
    preci_train_cla = precision_score(X_train_pre, Y_train, average=None)
    preci_test_all = precision_score(X_test_pre, Y_test, average='weighted')
    preci_test_cla = precision_score(X_test_pre, Y_test, average=None)

    recall_all = recall_score(x_pre, y, average='weighted')
    recall_cla = recall_score(x_pre, y, average=None)
    recall_train_all = recall_score(X_train_pre, Y_train, average='weighted')
    recall_train_cla = recall_score(X_train_pre, Y_train, average=None)
    recall_test_all = recall_score(X_test_pre, Y_test, average='weighted')
    recall_test_cla = recall_score(X_test_pre, Y_test, average=None)
    if r[2] != 0:
        preci_verify_all = precision_score(X_verify_pre, Y_verify, average='weighted')
        preci_verify_cla = precision_score(X_verify_pre, Y_verify, average=None)
        recall_verify_all = recall_score(X_verify_pre, Y_verify, average='weighted')
        recall_verify_cla = recall_score(X_verify_pre, Y_verify, average=None)
    else:
        preci_verify_all = 0.00
        preci_verify_cla = 0.00
        recall_verify_all = 0.00
        recall_verify_cla = 0.00
    res = {
        '总体精确率': preci_all,
        '分类总体精确率': preci_cla,
        '训练精确率': preci_train_all,
        '分类训练精确率': preci_train_cla,
        '测试精确率': preci_test_all,
        '分类测试精确率': preci_test_cla,
        '验证精确率': preci_verify_all,
        '分类验证精确率': preci_verify_cla,
        '总体召回率': recall_all,
        '分类总体召回率': recall_cla,
        '训练召回率': recall_train_all,
        '分类训练召回率': recall_train_cla,
        '测试召回率': recall_test_all,
        '分类测试召回率': recall_test_cla,
        '验证召回率': recall_verify_all,
        '分类验证召回率': recall_verify_cla
    }
    return res


def adaboost(data, coef, r,dic):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    y = y.astype(str).map(dic)
    r = list(map(int, r))
    if r[2] != 0:
        X, X_verify, Y, Y_verify = train_test_split(
            x, y, test_size=r[2] / sum(r), random_state=1, shuffle=True, stratify=y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=Y)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=y)
    # clf = AdaBoostClassifier(base_estimator='CART', n_estimators=12, learning_rate=1.0, algorithm='SAMME',
    #                          random_state=None)
    clf = AdaBoostClassifier(algorithm = coef[0],base_estimator = coef[1],n_estimators = int(coef[2]),
                             learning_rate = float(coef[3]),random_state = coef[4])
    clf.fit(X_train, Y_train)
    x_pre = clf.predict(x)
    X_train_pre = clf.predict(X_train)
    X_test_pre = clf.predict(X_test)
    if r[2] != 0:
        X_verify_pre = clf.predict(X_verify)
    preci_all = precision_score(x_pre, y, average='weighted')
    preci_cla = precision_score(x_pre, y, average=None)
    preci_train_all = precision_score(X_train_pre, Y_train, average='weighted')
    preci_train_cla = precision_score(X_train_pre, Y_train, average=None)
    preci_test_all = precision_score(X_test_pre, Y_test, average='weighted')
    preci_test_cla = precision_score(X_test_pre, Y_test, average=None)

    recall_all = recall_score(x_pre, y, average='weighted')
    recall_cla = recall_score(x_pre, y, average=None)
    recall_train_all = recall_score(X_train_pre, Y_train, average='weighted')
    recall_train_cla = recall_score(X_train_pre, Y_train, average=None)
    recall_test_all = recall_score(X_test_pre, Y_test, average='weighted')
    recall_test_cla = recall_score(X_test_pre, Y_test, average=None)
    if r[2] != 0:
        preci_verify_all = precision_score(X_verify_pre, Y_verify, average='weighted')
        preci_verify_cla = precision_score(X_verify_pre, Y_verify, average=None)
        recall_verify_all = recall_score(X_verify_pre, Y_verify, average='weighted')
        recall_verify_cla = recall_score(X_verify_pre, Y_verify, average=None)
    else:
        preci_verify_all = 0.00
        preci_verify_cla = 0.00
        recall_verify_all = 0.00
        recall_verify_cla = 0.00
    res = {
        '总体精确率': preci_all,
        '分类总体精确率': preci_cla,
        '训练精确率': preci_train_all,
        '分类训练精确率': preci_train_cla,
        '测试精确率': preci_test_all,
        '分类测试精确率': preci_test_cla,
        '验证精确率': preci_verify_all,
        '分类验证精确率': preci_verify_cla,
        '总体召回率': recall_all,
        '分类总体召回率': recall_cla,
        '训练召回率': recall_train_all,
        '分类训练召回率': recall_train_cla,
        '测试召回率': recall_test_all,
        '分类测试召回率': recall_test_cla,
        '验证召回率': recall_verify_all,
        '分类验证召回率': recall_verify_cla
    }
    return res


def desicion_tree(data, coef, r,dic,d):
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    y = y.astype(str).map(dic)
    r = list(map(int, r))
    if r[2] != 0:
        X, X_verify, Y, Y_verify = train_test_split(
            x, y, test_size=r[2] / sum(r), random_state=1, shuffle=True, stratify=y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=Y)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            x, y, test_size=r[1] / (r[0] + r[1]), random_state=1, shuffle=True, stratify=y)
    clf = DecisionTreeClassifier(criterion = coef[0], splitter = coef[1], max_leaf_nodes = coef[2], max_depth = coef[3],
                                 min_impurity_decrease = float(coef[4]), min_samples_split =int(coef[5]),
                                 min_samples_leaf = int(coef[6]), random_state=coef[7],class_weight=d)
    clf.fit(X_train, Y_train)
    x_pre = clf.predict(x)
    X_train_pre = clf.predict(X_train)
    X_test_pre = clf.predict(X_test)
    if r[2] != 0:
        X_verify_pre = clf.predict(X_verify)
    preci_all = precision_score(x_pre, y, average='weighted')
    preci_cla = precision_score(x_pre, y, average=None)
    preci_train_all = precision_score(X_train_pre, Y_train, average='weighted')
    preci_train_cla = precision_score(X_train_pre, Y_train, average=None)
    preci_test_all = precision_score(X_test_pre, Y_test, average='weighted')
    preci_test_cla = precision_score(X_test_pre, Y_test, average=None)

    recall_all = recall_score(x_pre, y, average='weighted')
    recall_cla = recall_score(x_pre, y, average=None)
    recall_train_all = recall_score(X_train_pre, Y_train, average='weighted')
    recall_train_cla = recall_score(X_train_pre, Y_train, average=None)
    recall_test_all = recall_score(X_test_pre, Y_test, average='weighted')
    recall_test_cla = recall_score(X_test_pre, Y_test, average=None)
    if r[2] != 0:
        preci_verify_all = precision_score(X_verify_pre, Y_verify, average='weighted')
        preci_verify_cla = precision_score(X_verify_pre, Y_verify, average=None)
        recall_verify_all = recall_score(X_verify_pre, Y_verify, average='weighted')
        recall_verify_cla = recall_score(X_verify_pre, Y_verify, average=None)
    else:
        preci_verify_all = 0.00
        preci_verify_cla = 0.00
        recall_verify_all = 0.00
        recall_verify_cla = 0.00
    res = {
        '总体精确率': preci_all,
        '分类总体精确率': preci_cla,
        '训练精确率': preci_train_all,
        '分类训练精确率': preci_train_cla,
        '测试精确率': preci_test_all,
        '分类测试精确率': preci_test_cla,
        '验证精确率': preci_verify_all,
        '分类验证精确率': preci_verify_cla,
        '总体召回率': recall_all,
        '分类总体召回率': recall_cla,
        '训练召回率': recall_train_all,
        '分类训练召回率': recall_train_cla,
        '测试召回率': recall_test_all,
        '分类测试召回率': recall_test_cla,
        '验证召回率': recall_verify_all,
        '分类验证召回率': recall_verify_cla
    }

    return res


def algo_times(data, coef, r, time, method,dic,d):
    f = pd.DataFrame()
    coef = coef_format(method,coef)
    if method == 'dct':
        for i in range(time):
            af = pd.DataFrame(data=desicion_tree(data, coef, r,dic,d))
            f = pd.concat((f, af))
    elif method == 'ran':
        for i in range(time):
            for i in range(time):
                af = pd.DataFrame(data=random_forest(data, coef, r,dic,d))
                f = pd.concat((f, af))
    elif method == 'ada':
        for i in range(time):
            af = pd.DataFrame(data=adaboost(data, coef, r,dic))
            f = pd.concat((f, af))
    elif method == 'svm':
        for i in range(time):
            af = pd.DataFrame(data=svm_classify(data, coef, r,dic,d))
            f = pd.concat((f, af))

    return f
