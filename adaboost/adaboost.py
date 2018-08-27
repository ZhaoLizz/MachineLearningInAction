
import numpy as np

def load_simp_data():
    data_mat = np.matrix([[1.0, 2.1],
                          [2.0, 1.1],
                          [1.3, 1.0],
                          [1.0, 1.0],
                          [2.0, 1.0]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels

def test():
    print()
    data_mat,class_labels = load_simp_data()
    print(data_mat)
    print(class_labels)

if __name__ == '__main__':
    test()