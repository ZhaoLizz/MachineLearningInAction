import operator
import numpy as np
from os import listdir
from collections import Counter


def create_data_set():
    group = np.array([1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, data_set, labels, k):
    """

    :param inX:输入向量
    :param data_set: 训练样本
    :param labels: 标签向量
    :param k: 最近邻的数目
    """

    # 方法一 -----------------------------------------------------------------
    data_set_size = data_set.shape[0]  # 数据集大小
    # 把输入向量广播为数据集大小行,相当于对每行数据集数据都计算一次各坐标分量的差值
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set
    """
    tile用法:
    inx[1,2,3]
    
    In [8]: tile(inx, (3, 1))
    Out[8]:
    array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
        
         In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    sq_diff_mat = diff_mat ** 2  # 先做差,然后每项的差平方
    sq_distance = sq_diff_mat.sum(axis=1)  # 横向相加
    distances = sq_distance ** 0.5  # 开方
    sorted_dist_indicies = distances.argsort()  # 返回数组从小到大的索引值
    class_count = {}  # label名 :  出现次数
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1  # 在字典中将该类型加1,get传入的第二个参数是返回value的默认值

    # 排序并返回出现次数最多的类型
    # 字典的items()方法以列表返回可遍历的(键,值)元祖数组
    """
    >>>dict = {'Name': 'Zara', 'Age': 7} 
    >>>dict.items()
    dict_items([('Name', 'Zara'), ('Age', 7)])
    """
    # 第二个参数key=operator.itemgetter(1)是先比较第几个元素,传1就是比较第二个元素,也就是出现次数
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def classify1(inx, data_set, labels, k):
    # 计算距离: 先利用broadcasting求数据集每行和输入向量的差值,然后计算欧式距离
    # 这里求和: 每行的值横向相加求和,返回一个列向量

    # print(inx.shape)
    # print(data_set.shape)
    # print((inx - data_set).shape)

    dist = np.sum((inx - data_set) ** 2, axis=1) ** 0.5
    # 找到前k个最近的标签
    k_labels = [labels[index] for index in dist.argsort()[0:k]]  # 由小到大排序
    # 使用collections.Counter同济各个标签出现的次数
    # most_common返回出现次数最多的标签tuple,如[('lable1', 2)]
    label = Counter(k_labels).most_common(1)[0][0]
    return label


def file2matrix(filename):
    fr = open(filename, 'r')
    number_of_lines = len(fr.readlines())
    return_mat = np.zeros((number_of_lines, 3))  # 3列代表三个特征
    class_label_vector = []
    index = 0
    fr = open(filename, 'r')
    for line in fr.readlines():
        line = line.strip()  # 移除字符串头尾的参数值(这里是空格)
        list_from_line = line.split('\t')
        return_mat[index] = list_from_line[0:3]  # 每一列的属性
        class_label_vector.append(int(list_from_line[-1]))  # 属性
        index += 1
    return return_mat, class_label_vector


def auto_norm(dataset):
    """
    归一化特征
    Y = (X - Xmin) / (Xmax - Xmin)
    """
    min_value = dataset.min(0)
    max_value = dataset.max(0)
    ranges = max_value - min_value
    norm_dataset = (dataset - min_value) / ranges
    return norm_dataset, ranges, min_value


def dating_class_test():
    """
    对约会网站测试
    :return:
    """
    ho_ratio = 0.1  # 测试集比例
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_values = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]  # 矩阵行数
    num_test_vecs = int(m * ho_ratio)  # 测试集样本数量
    error_count = 0

    for i in range(num_test_vecs):  # 交叉验证
        classifier_result = classify1(norm_mat[i], norm_mat[num_test_vecs:m], dating_labels[num_test_vecs:m], 3)
        print('the lcassifier came bakc with : %d, the read answer is : %d' % (classifier_result, dating_labels[i]))
        error_count += classifier_result != dating_labels[i]
    print('the total error rate is %f ' % (error_count / num_test_vecs))
    print(error_count)


if __name__ == '__main__':
    dating_class_test()
