import numpy as np


class Perceptron(object):
    """
    Perceptron classifier

    Parameters:
    ------------
    eta:float
        Learning rate(bbetween 0.0 and 1.0)
    n_iter:int
        Passes o
        ver the training dataset

    Attributes
    ------------
    w_: 1d-array
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        """
        Calculate net input
        return y_i
        """
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)  # 三目条件运算

    def fit(self, X, y):
        """
        Fit training data
        :param X: array-like,shape = [n_samples,n_features]
        :param y: array-list,shape = [n_samples]
        :return: self:object
        """
        self.w_ = np.zeros(1 + X.shape[1])  # 一维向量，加上b = w_0，所以是X的列数加一
        self.errors_ = []   # 记录每次迭代的误差个数,下标是迭代次数

        for _ in range(self.n_iter):  # 迭代n_iter次
            errors = 0
            #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            # 随机梯度下降法: 每次仅仅使用一个样本更新参数
            for xi, target in zip(X, y):
                # 如果target和predict值相同,预测正确,updata就等于0,否则等于+-2
                update = self.eta * (target - self.predict(xi))  # eta * y_i
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            print(errors)
            self.errors_.append(errors)
        return self
