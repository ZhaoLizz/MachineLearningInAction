# -*- coding: UTF-8 -*-
import numpy as np
from collections import Counter
import decision_tree_plot as dt_plot

def calc_shannon_ent(dataSet):
    label_count = Counter(data[-1] for data in dataSet)  # 返回一个字典,{label : counts}
    probs = [p[1] / len(dataSet) for p in label_count.items()]  # items把所有k:v组成元组,返回元组的list
    shannonEnt = sum([-p * np.log2(p) for p in probs])
    return shannonEnt


def create_data_set():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return np.array(dataSet), np.array(labels)


def split_data_set(data_set, index, value):
    """
    找出第index列的值为value的行
    """
    ret_data_set = []
    # index列的值为value的行
    mask = data_set[:, index] == value
    # 去掉index列
    return np.delete(data_set[mask], index, axis=1)


def choose_best_feature_to_split(data_set):
    """
    :param data_set:
    :return:
    """
    num_features = len(data_set[0]) - 1  # 特征的数量
    base_entropy = calc_shannon_ent(data_set)  # 数据集的熵
    best_info_gain = 0.0  # 信息增益
    best_feature = -1
    # 遍历每个特征
    for i in range(num_features):
        feat_list = data_set[:, i]  # 第i列
        unique_vals = set(feat_list)  # 第i个特征的每种取值
        new_emtropy = 0.0
        # 遍历i特征的每个取值,每个特征对应一个条件熵
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)  # 第i个特征值为value的子数据集
            prob = len(sub_data_set) / len(data_set)
            new_emtropy += prob * calc_shannon_ent(sub_data_set)
        infogain = base_entropy - new_emtropy
        if infogain > best_info_gain:
            best_feature = i
            best_info_gain = infogain
    return best_feature


def majority_cnt(class_list):
    """
    选择出现次数最多的label值用于计算树的返回值
    :param class_list: y向量
    """
    major_label = Counter(class_list).most_common(1)[0]  # Return a list of the n most common elements and their counts
    return major_label


def create_tree(data_set, labels):
    """
    :param labels:特征名字的向量
    """
    class_list = [example[-1] for example in data_set]
    # 如果当前结点label全是一种值
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果特征向量的len==1,也就是特征值已经被删除完了,只剩一个最后的label了
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]  # 特征下标对应的特征的名字
    tree = {best_feat_label: {}}
    labels = np.delete(labels, best_feat)

    # 根据当前最优特征的每种取值对当前数据集进行划分
    feat_values = [example[best_feat] for example in data_set]  # dataset的best_feat列
    unique_vals = set(feat_values)
    # 对于最优特征的每种取值
    for value in unique_vals:
        sub_labels = labels[:]  # 剩余的标签名字
        tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return tree

def classify(input_tree,feat_labels,test_vec):
    """
    对新数据分类
    :param input_tree: 训练好的决策树模型
    :param feat_labels: label名字向量
    :param test_vec: 测试数据
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    # index() 函数用于从列表中找出某个值第一个匹配项的索引位置。
    feat_index = feat_labels.index(first_str) #当前结点label对应的下标
    key = test_vec[feat_index] # 测试数据对应当前结点的label值
    value_of_feat = second_dict[key] # 树模型中,测试数据的对应这个结点的label值对应的子树
    print("+++",first_str,'xxx',second_dict,'---',key,'>>>',value_of_feat)
    # 判断分支是否结束
    if isinstance(value_of_feat,dict):
        class_label = classify(value_of_feat,feat_labels,test_vec)
    else:
        class_label = value_of_feat
    return value_of_feat


def store_tree(tree,filename):
    import pickle
    with open(filename,'wb') as fw:
        pickle.dump(tree,fw)

def grab_tree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def contact_lenses_test():
    print('gherafdsdeasf')
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels = ['age','prescrip','astigmatic','tearRate']
    lenses_tree = create_tree(lenses,lenses_labels)
    print(lenses_tree)

contact_lenses_test()
