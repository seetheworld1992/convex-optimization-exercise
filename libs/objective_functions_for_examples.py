import numpy as np
from abc import ABCMeta, abstractmethod


class ObjectiveFunction(metaclass=ABCMeta):
    """
    目的関数の抽象クラス
    """

    @abstractmethod
    def objective_func(self, x : np.ndarray):
        pass

    @abstractmethod
    def full_path_grad(self, x : np.ndarray, feat_matrix : np.ndarray, label_vec : np.ndarray):
        pass

    @abstractmethod
    def one_path_grad(self, x : np.ndarray, feat_vec : np.ndarray, label_int : int):
        pass

    @abstractmethod
    def hessian_matrix(self, x : np.ndarray, feat_matrix : np.ndarray, label_vec : np.ndarray):
        pass


class NonQuadraticExample1(ObjectiveFunction):
    """
    二次元ベクトルの関数
    Boyd and Vandenberghe, Convex Optimization, pp.470 (9.20)
    """

    def __init__(self):
        pass

    def objective_func(self, x):
        return np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(- x[0] - 0.1)

    def full_path_grad(self, x):
        grad = np.zeros(2)
        grad[0] = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(- x[0] - 0.1)
        grad[1] = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)

        return grad

    def one_path_grad(self):
        return 0

    def hessian_matrix(self, x):
        hessian = np.zeros((2,2))
        hessian[0][0] = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(- x[0] - 0.1)
        hessian[0][1] = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
        hessian[1][0] = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
        hessian[1][1] = 9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)
        
        return hessian


class LinearPlusLogBarrier(ObjectiveFunction):
    """
    線形 + 対数バリア 関数
    """

    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c        

    def objective_func(self, x):
        return np.dot(self.c, x) - np.sum(np.log( self.b - np.dot(self.A, x ))) 

    def full_path_grad(self, x):
        m = self.A.shape[0]
        n = self.c.shape[0]        
        res = np.zeros(n)        
        for i in range(m):
            ai = self.A[i,:]
            res = res + ai / (self.b[i] - np.dot(ai, x))        
        res = self.c + res

        return res

    def one_path_grad(self):
        return 0

    def hessian_matrix(self, x):
        m = self.A.shape[0]
        n = self.c.shape[0]        
        hessian = np.zeros((n,n))    
        for i in range(m):
            ai = self.A[i,:]
            hi = np.dot(ai, x) - self.b[i]
            hessian = hessian + np.outer(ai, ai) / (hi * hi)

        return hessian




if __name__ == "__main__":

    x = np.array([[1, -1 ,2], [0 , 1 ,0]])
    print(x[1,:])
    pass