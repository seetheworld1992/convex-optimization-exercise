import numpy as np
from abc import ABCMeta, abstractmethod


class StepSizeRule(metaclass=ABCMeta):
    """
    ステップサイズルールの抽象クラス
    当面は劣勾配法だけに用いる想定

    """
    @abstractmethod
    def step_size_rule(self, x : np.ndarray, y : np.ndarray):
        pass


class ConstantStepSize(StepSizeRule):
    """
    具象クラス : 定数ステップサイズ

    Attributes:
        後で書く

    """

    def __init__(self, h):
        """
        コンストラクタ
        Args:
            h : float64                
        """ 
        self.h =  h

    def step_size_rule(self, x, y = 0):

        return self.h


class ConstantStepLength(StepSizeRule):
    """
    具象クラス : 定数ステップ長

    Attributes:
        後で書く

    """

    def __init__(self, h):
        """
        コンストラクタ
        Args:
            h : float64                
        """ 
        self.h =  h

    def step_size_rule(self, x, y = 0):

        return self.h / np.linalg.norm(x, ord = 2)


class NonsummableDiminishing1(StepSizeRule):
    """
    具象クラス : 非総和可能減少その1

    Attributes:
        後で書く

    """

    def __init__(self, h):
        """
        コンストラクタ
        Args:
            h : float64                
        """ 
        self.h =  h

    def step_size_rule(self, x, iteration, y = 0):

        return self.h / np.sqrt(iteration + 1)


class NonsummableDiminishing2(StepSizeRule):
    """
    具象クラス : 非総和可能減少その2

    Attributes:
        後で書く

    """

    def __init__(self, h):
        """
        コンストラクタ
        Args:
            h : float64                
        """ 
        self.h =  h

    def step_size_rule(self, x, iteration, y = 0):

        return self.h / (iteration + 1)


class StepSizeRuleForObfgs(StepSizeRule):
    """
    具象クラス :     確率的準ニュートン法 BFGS のアルゴリズム
                     Nicol N. Schraudolph, Jin Yu, Simon G¨unter の
                     A Stochastic Quasi-Newton Method for Online Convex Optimization の(7)式
                     http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf

    Attributes:
        後で書く

    """

    def __init__(self, h, tau):
        """
        コンストラクタ
        Args:
            h : float64                
        """ 
        self.h =  h
        self.tau =  tau

    def step_size_rule(self, x, iteration, y = 0):

        return self.tau / (self.tau + iteration) * self.h


if __name__ == "__main__":
    print(np.sqrt(2))
    pass