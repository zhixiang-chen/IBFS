"""This file contains methods and objects which are reused through multiple files.
    这个文件包含：
        常量：Falcon签名中使用的整数模q
        函数：split(f)将多项式f分割为两个子多项式
             merge(f_list)将两个多项式合并为一个多项式
             sqnorm(v)计算向量v的平方欧几里得范数
"""


"""q is the integer modulus which is used in Falcon.
    q为在Falcon签名中使用的整数模
"""
q = 12 * 1024 + 1


def split(f):
    """Split a polynomial f in two polynomials.

    Args:
        f: a polynomial

    Format: coefficient
    将一个多项式f分割为两个多项式
    """
    n = len(f) # 获取多项式 f 的长度（系数的个数）
    # 将多项式f分割成两个子多项式
    f0 = [f[2 * i + 0] for i in range(n // 2)] # f0包含偶数索引的系数
    f1 = [f[2 * i + 1] for i in range(n // 2)] # f1包含奇数索引的系数
    return [f0, f1] # 返回包含两个子多项式f0和f1的列表


def merge(f_list):
    """Merge two polynomials into a single polynomial f.

    Args:
        f_list: a list of polynomials

    Format: coefficient
    将两个子多项式合并为一个多项式
    """
    f0, f1 = f_list # 从列表中获取两个子多项式f0和f1
    n = 2 * len(f0) # 计算合并后的多项式长度
    f = [0] * n # 创建一个长度为2 * len(f0)的零列表f
    # 将f0和f1的系数合并到f中
    for i in range(n // 2):
        f[2 * i + 0] = f0[i] # 将 f0 中的系数放到 f 的偶数位置
        f[2 * i + 1] = f1[i] # 将 f1 中的系数放到 f 的奇数位置
    return f # 返回合并后的多项式


def sqnorm(v):
    """Compute the square euclidean norm of the vector v.
        用于计算向量v的平方欧几里得范数
    """
    res = 0 # 初始化结果变量 res，用于存储范数的平方和
    for elt in v: # 遍历向量 v 中的每个元素 elt
        for coef in elt: # 遍历元素 elt 中的每个系数 coef
            res += coef ** 2 # 将 coef 的平方加到 res 中
    return res # 返回计算得到的平方欧几里得范数
