import numpy as np


def load_simp_data():
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    """
    单层决策树(树桩)
    对于所选的特征列,大于和小于阈值的根据ineq参数分别设置为+-1
    :param data_matrix:
    :param dimen: 所选特征的下标
    :param thresh_val: 特征列要比较的值
    :param thresh_ineq: 针对阈值的取值方式
    :return:
    """
    # 首先将返回数组的全部元素设置为1,然后把不满足阈值的设置为-1
    ret_array = np.ones((np.shape(data_matrix)[0], 1))  # (N,1)  决策分类结果
    if thresh_ineq == 'lt':
        # 取数据集dimen列,把这一列<= thresh_Val的值都改为-1.0(>阈值的值仍为初始值1)
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, D):
    """
    生成单层决策树模型
    :param data_arr:
    :param class_labels:
    :param D: 最初的特征权重向量
    :return:
        best_stump : 最优的分类器模型
        min_error : 错误率
        best_class_est : 预测结果
    """
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T

    m, n = np.shape(data_mat)
    num_steps = 100.0
    best_stump = {}  # 最优单层决策树的相关信息
    best_class_est = np.mat(np.zeros((m, 1)))
    min_err = np.inf
    # 遍历每个特征
    for i in range(n):
        range_min = data_mat[:, i].min()  # 当前特征的最大最小值
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps  # 决策树桩的阈值每次迭代 挪动的距离
        # 遍历每个树桩的阈值
        for j in range(-1, int(num_steps) + 1):  # 含前不含后
            # 对于两种左右分类方式
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)  # 决策阈值
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)  # 树桩决策结果
                # 先初始化error array为1,然后把分类正确的置位0
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_err = D.T * err_arr  # mat  dot, 加权计算分类误差率
                # print('split: dim{} , thresh {} , thresh inequal : {},the weighted err is {}'.format(i, thresh_val,
                #                                                                                      inequal,
                #                                                                                      weighted_err))
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i  # 最佳分类方式对应的特征
                    best_stump['thresh'] = thresh_val  # 树桩分界阈值
                    best_stump['ineq'] = inequal  # 左右分类方式
    return best_stump, min_err, best_class_est


def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    """
    :return:
        weak_class_arr 弱分类器集合
        agg_class_est 预测的分类结果值
    """
    weak_class_arr = []  # 弱分类器list
    m = np.shape(data_arr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化训练数据集的权重
    agg_class_est = np.mat(np.zeros((m, 1)))  # 预测结果
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)  # 决策树模型
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # max为了避免分母为0
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)  # 每轮迭代生成一个弱分类器

        # print('class_est: {}'.format(class_est.T))  # 预测结果
        # 计算规范化因子Zm,更新权值分布
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)  # multiply对应位置广播相乘,而不是dot
        D = np.multiply(D, np.exp(expon))
        D = D / np.sum(D)
        agg_class_est += alpha * class_est  # 对基本分类器线性组合
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print('total error : {}\n'.format(error_rate))
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def ada_classify(data_to_class, classifier_arr):
    """
    利用训练好的模型分类数据
    :param data_to_class:
    :param classifier_arr: 弱分类器列表(dict)
    :return: +1 or -1
    """
    data_mat = np.mat(data_to_class)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    # 对于每个弱分类器(iter轮迭代生成的i个弱分类器)
    for i in range(len(classifier_arr)):
        class_est = stump_classify(
            data_mat,
            classifier_arr[i]['dim'],
            classifier_arr[i]['thresh'],
            classifier_arr[i]['ineq']
        )
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        # print(agg_class_est)
    return np.sign(agg_class_est)


def plot_roc(pred_strengths, class_labels):
    """
    打印ROC曲线,并计算AUC的面积大小
    :param pred_strengths: 最终预测结果(可信度)
    :param class_labels: y
    :return:
    """
    import matplotlib.pyplot as plt
    y_sum = 0.0
    # 对label中的正样本求和
    num_pos_class = np.sum(np.array(class_labels) == 1.0)
    y_step = 1 / float(num_pos_class)
    x_step = 1 / float(len(class_labels) - num_pos_class)
    sorted_indicies = pred_strengths.argsort()  # 预测值的确信度从小到大的索引值
    """
    example: array([[4, 0, 7, 1, 2, 6, 3, 5]], dtype=int64)
    """

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    cur = (1.0, 1.0)  # cursor光标值
    # 按照确信度由小到大的顺序
    for index in sorted_indicies.tolist()[0]:  # tolist()[0]的结果就是索引值一维列表
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]

        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    # 画对角线的虚线
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title("ROC curve for AdaBoost horse colic detection system")
    # 设置图的坐标区间
    ax.axis([0,1,0,1])
    plt.show()

    print("the Area Under the Curve is: ", y_sum * x_step)



def load_dataset(filename):
    num_feat = len(open(filename).readline().split('\t'))  # 特征数量
    data_arr = []
    label_arr = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []  # 当前行去掉最后一列(即去掉label)剩下的
        cur_line = line.strip().split('\t')  # 当前行的列表
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))
    return np.matrix(data_arr), label_arr


def test():
    print()
    data_mat, class_labels = load_dataset('horseColicTraining2.txt')
    print('training data: ', data_mat.shape, len(class_labels))
    # 弱分类器列表,预测结果
    weak_class_arr, agg_class_est = ada_boost_train_ds(data_mat, class_labels, num_it=40)
    plot_roc(agg_class_est,class_labels)

    # 加载测试集,在训练集训练好的模型上预测结果
    test_arr, test_label_arr = load_dataset('horseColicTest2.txt')
    prediction10 = ada_classify(test_arr, weak_class_arr)
    err_arr = np.mat(np.ones((67, 1)))
    error_rate = err_arr[prediction10 != np.mat(test_label_arr).T].sum() / 67
    print('test error rate:', error_rate)


if __name__ == '__main__':
    test()
