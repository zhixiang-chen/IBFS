"""
    实现离散高斯分布采样，使用了半高斯分布和拒绝采样
"""

# Importing dependencies
from math import floor

# Use high-quality randomness
# The "secrets" library could also work (Python >= 3.6)
from os import urandom


# Upper bound on all the values of sigma
# 定义高斯分布的标准差上限
MAX_SIGMA = 1.8205
INV_2SIGMA2 = 1 / (2 * (MAX_SIGMA ** 2))

# Precision of RCDT
RCDT_PREC = 72 # RCDT（反向累积分布表）的精度，表示使用的位数

# ln(2) and 1 / ln(2), with ln the natural logarithm
LN2 = 0.69314718056 # 自然对数 ln(2) 的值
ILN2 = 1.44269504089 # ln(2) 的倒数值，等于 1 / ln(2)


# RCDT is the reverse cumulative distribution table of a distribution that
# is very close to a half-Gaussian of parameter MAX_SIGMA.
# 反向累积分布表，用于从半高斯分布中采样。每个元素表示一个阈值，用于快速查找采样值
RCDT = [
    3024686241123004913666,
    1564742784480091954050,
    636254429462080897535,
    199560484645026482916,
    47667343854657281903,
    8595902006365044063,
    1163297957344668388,
    117656387352093658,
    8867391802663976,
    496969357462633,
    20680885154299,
    638331848991,
    14602316184,
    247426747,
    3104126,
    28824,
    198,
    1]


# C contains the coefficients of a polynomial that approximates exp(-x)
# More precisely, the value:
# (2 ** -63) * sum(C[12 - i] * (x ** i) for i in range(i))
# Should be very close to exp(-x).
# This polynomial is lifted from FACCT: https://doi.org/10.1109/TC.2019.2940949
# 多项式系数，用于近似计算 exp(-x)，这些系数用于 `approxexp` 函数中
C = [
    0x00000004741183A3,
    0x00000036548CFC06,
    0x0000024FDCBF140A,
    0x0000171D939DE045,
    0x0000D00CF58F6F84,
    0x000680681CF796E3,
    0x002D82D8305B0FEA,
    0x011111110E066FD0,
    0x0555555555070F00,
    0x155555555581FF00,
    0x400000000002B400,
    0x7FFFFFFFFFFF4800,
    0x8000000000000000]


def basesampler(randombytes=urandom):
    """
    Sample z0 in {0, 1, ..., 18} with a distribution
    very close to the half-Gaussian D_{Z+, 0, MAX_SIGMA}.
    Takes as (optional) input the randomness source (default: urandom).
    """
    # 从随机字节中生成一个整数 u
    u = int.from_bytes(randombytes(RCDT_PREC >> 3), "little")

    z0 = 0 # 初始化采样值为 0
    for elt in RCDT: # 遍历反向累积分布表（RCDT），逐步累加
        z0 += int(u < elt) # 如果 u 小于当前的表值，则 z0 增加 1
    return z0 # 返回采样结果


def approxexp(x, ccs):
    """
    Compute an approximation of 2^63 * ccs * exp(-x).

    Input:
    - a floating-point number x
    - a scaling factor ccs
    Both inputs x and ccs MUST be positive.

    Output:
    - an integral approximation of 2^63 * ccs * exp(-x).

    用于计算2^63 * ccs * exp(-x)的近似值，ccs是一个缩放因子，x是输入
    """
    # y, z are always positive
    y = C[0] # 初始化多项式系数 y 为第一个系数
    # Since z is positive, int is equivalent to floor
    z = int(x * (1 << 63)) # 将 x 扩展为固定精度
    for elt in C[1:]: # 遍历多项式的其他系数，逐步计算多项式的值
        y = elt - ((z * y) >> 63) # 使用多项式进行逼近
    z = int(ccs * (1 << 63)) << 1 # 将 ccs 转换为固定精度
    y = (z * y) >> 63 # 调整结果的精度
    return y # 返回逼近的结果


def berexp(x, ccs, randombytes=urandom):
    """
    Return a single bit, equal to 1 with probability ~ ccs * exp(-x).
    Both inputs x and ccs MUST be positive.
    Also takes as (optional) input the randomness source (default: urandom).

    伯努利采样，该函数返回一个单独的比特，概率约为ccs×exp(-x)
    """
    s = int(x * ILN2) # 计算 x 的整数部分，s 约等于 log(x) / ln(2)
    r = x - s * LN2 # 计算 x 的小数部分
    s = min(s, 63) # 限制 s 的最大值为 63
    z = (approxexp(r, ccs) - 1) >> s # 使用 exp 函数计算，得到的结果右移 s 位
    for i in range(56, -8, -8): # 通过随机字节进一步调整结果
        p = int.from_bytes(randombytes(1), "little") # 获取一个随机字节
        w = p - ((z >> i) & 0xFF) # 从字节中提取部分数据
        if w:
            break # 一旦有差异，退出循环
    return (w < 0) # 返回一个 0 或 1 的结果


def samplerz(mu, sigma, sigmin, randombytes=urandom):
    """
    Given floating-point values mu, sigma (and sigmin),
    output an integer z according to the discrete
    Gaussian distribution D_{Z, mu, sigma}.

    Input:
    - the center mu
    - the standard deviation sigma
    - a scaling factor sigmin
    - optional: the randomness source randombytes (default: urandom)
      randombytes(k) should output k pseudorandom bytes
    The inputs MUST verify 1 < sigmin < sigma < MAX_SIGMA.

    Output:
    - a sample z from the distribution D_{Z, mu, sigma}.

    离散高斯分布采样器
    """
    s = int(floor(mu)) # 取 mu 的整数部分
    r = mu - s # 计算 mu 的小数部分
    dss = 1 / (2 * sigma * sigma) # 计算高斯分布的归一化常数
    ccs = sigmin / sigma # 计算缩放因子

    # 不断采样直到接受一个样本
    while(1):
        # Sampler z0 from a Half-Gaussian
        z0 = basesampler(randombytes=randombytes) # 从半高斯分布中采样
        # Convert z0 into a pseudo-Gaussian sample z
        # 使用 z0 生成一个伪高斯样本 z
        b = int.from_bytes(randombytes(1), "little") # 获取一个随机字节
        b &= 1 # 取最低有效位
        z = b + (2 * b - 1) * z0 # 生成最终的 z 样本
        # Rejection sampling to obtain a true Gaussian sample
        # 拒绝采样过程
        x = ((z - r) ** 2) * dss # 计算拒绝采样的标准值
        x -= (z0 ** 2) * INV_2SIGMA2 # 修正误差
        if berexp(x, ccs, randombytes=randombytes): # 使用伯努利分布接受或拒绝样本
            return z + s # 返回最终的采样值
