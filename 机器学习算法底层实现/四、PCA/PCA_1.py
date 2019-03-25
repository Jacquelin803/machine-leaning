import numpy as np

class PCA:
    
    def __init__(self, n_components):
        assert n_components >= 1, "n_components must be valid"
        self.n_components=n_components
        self.components_ = None

    def fit(self,X,eta=0.01,n_iters=1e4):
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        #数据均值归0
        def demean(X):
            return X-np.mean(X,axis=0)

        # f函数定义
        def f(w, X):
            return np.sum ((X.dot (w)) ** 2) / len (X)

        # 梯度正规计算
        def df(w, X):
            return X.T.dot (X.dot (w)) * 2 / len (X)

        # 每次都要将w长度归为1
        def direction(w):
            return w / np.linalg.norm (w)

        # 梯度上升函数定义（每次都要用方向向量，长度必须为1）
        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
            w = direction (initial_w)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = df (w, X)
                last_w = w
                w = w + eta * gradient
                w = direction (w)
                if (abs (f (w, X) - f (last_w, X)) < epsilon):
                    break
                cur_iter += 1
            return w

        X_pca=demean(X)
        self.components_=np.empty(shape=(self.n_components,X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random (X_pca.shape[1])
            w = first_component (X_pca, initial_w, eta,n_iters)
            self.components_[i,:]=w

            X_pca = X_pca - X_pca.dot(w).reshape (-1, 1) * w
        return self

    def transform(self,X):
        """降维：将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self,X):
        """将低维显示在高维图像里：将给定的X，反向映射回原来的特征空间,X一般为上述求出的w"""
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return 'PCA(n_components=%d)' %self.n_components











