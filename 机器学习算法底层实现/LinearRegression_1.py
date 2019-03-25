from sklearn.metrics import r2_score
import numpy as np


class LinearRegression:

    def __init__(self):
        '''初始化'''
        self.coef_=None  #本包里计算的指标
        self.intercept_=None
        self._theta=None #私有指标，调用包的人不会知道

    def fit_normal(self,X_train,y_train):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])  #将列向量[1..]与X_train横向合并

        self._theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)   #theta公式计算，取逆，矩阵转置，矩阵乘

        self.intercept_=self._theta[0]
        self.coef_=self._theta[1:]

        return  self

    def fit_GD(self,X_train, y_train, eta=0.01, n_interp=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b=np.hstack([np.ones((len(X_train),)),X_train])   #方便计算，将X_train转化为X_b

        def J(theta,X_b,y):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(y)
            except:
                return float('inf')
        def dJ(theta,X_b,y):
            '''计算J函数的导数'''
            #res=np.empty(len(theta))
            #res[0]=np.sum(X_b.dot(theta)-y)   #计算theta0
            #for i in range(1,len(theta)):
             #   res[i]=(X_b.dot(theta)-y).dot(X_b[:,i])
            
            return X_b.T.dot(X_b.dot(theta)-y)*2/len(y)

        def Gradient_Descent(X_b,y,initial_theta, eta, n_interp=1e4, epsilon=1e-8):  # 初始值，下降比例，差值
            theta = initial_theta
            i_interp = 0    #运行次数
            while i_interp < n_interp:
                gradient = dJ (theta,X_b,y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs (J(theta,X_b,y) - J(last_theta,X_b,y)) < epsilon):
                    break
                i_interp += 1
            return theta

        initial_theta = np.zeros (X_b.shape[1])
        self._theta=Gradient_Descent(X_b,y_train,initial_theta,eta,n_interp)
        self.intercept_=self._theta[0]
        self.coef_=self._theta[1:]
        return self


    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b=np.hstack([np.ones((len(X_predict), 1)), X_predict])    #将预测原值转化为正确的矩阵
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict=self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return 'LinearRegression()'























