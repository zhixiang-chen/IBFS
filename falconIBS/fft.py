"""This file contains an implementation of the FFT.

The FFT implemented here is for polynomials in R[x]/(phi), with:
- The polynomial modulus phi = x ** n + 1, with n a power of two, n =< 1024

The code is voluntarily very similar to the code of the NTT.
It is probably possible to use templating to merge both implementations.

实现了多项式的快速傅里叶变换FFT和逆变换IFFT
"""

from falconIBS.common import split, merge         # Import split and merge
from falconIBS.fft_constants import roots_dict    # Import constants useful for the FFT


def split_fft(f_fft):
    """Split a polynomial f in two polynomials.

    Args:
        f: a polynomial

    Format: FFT

    Corresponds to algorithm 1 (splitfft_2) of Falcon's documentation.

    将f_fft分割为两个子多项式
    """
    n = len(f_fft)
    w = roots_dict[n] # 获取n次根
    f0_fft = [0] * (n // 2)
    f1_fft = [0] * (n // 2)
    for i in range(n // 2):
        f0_fft[i] = 0.5 * (f_fft[2 * i] + f_fft[2 * i + 1])
        f1_fft[i] = 0.5 * (f_fft[2 * i] - f_fft[2 * i + 1]) * w[2 * i].conjugate()
    return [f0_fft, f1_fft]


def merge_fft(f_list_fft):
    """Merge two or three polynomials into a single polynomial f.

    Args:
        f_list: a list of polynomials

    Format: FFT

    Corresponds to algorithm 2 (mergefft_2) of Falcon's documentation.

    将两个或三个多项式合并为一个多项式
    """
    f0_fft, f1_fft = f_list_fft
    n = 2 * len(f0_fft)
    w = roots_dict[n] # 获取 n 次根
    f_fft = [0] * n
    for i in range(n // 2):
        f_fft[2 * i + 0] = f0_fft[i] + w[2 * i] * f1_fft[i]
        f_fft[2 * i + 1] = f0_fft[i] - w[2 * i] * f1_fft[i]
    return f_fft


def fft(f):
    """Compute the FFT of a polynomial mod (x ** n + 1).

    Args:
        f: a polynomial

    Format: input as coefficients, output as FFT

    计算多项式f的快速傅里叶变换
    """
    n = len(f)
    if (n > 2):
        f0, f1 = split(f) # 将多项式分割为两个子多项式
        f0_fft = fft(f0) # 递归计算 f0 的 FFT
        f1_fft = fft(f1) # 递归计算 f1 的 FFT
        f_fft = merge_fft([f0_fft, f1_fft]) # 合并结果
    elif (n == 2):
        f_fft = [0] * n
        f_fft[0] = f[0] + 1j * f[1] # 计算 FFT
        f_fft[1] = f[0] - 1j * f[1] # 计算 FFT
    return f_fft


def ifft(f_fft):
    """Compute the inverse FFT of a polynomial mod (x ** n + 1).

    Args:
        f: a FFT of a polynomial

    Format: input as FFT, output as coefficients

    计算多项式的逆快速傅里叶变换
    """
    n = len(f_fft)
    if (n > 2):
        f0_fft, f1_fft = split_fft(f_fft) # 分割 FFT
        f0 = ifft(f0_fft) # 递归计算 f0 的 IFFT
        f1 = ifft(f1_fft) # 递归计算 f1 的 IFFT
        f = merge([f0, f1]) # 合并结果
    elif (n == 2):
        f = [0] * n
        f[0] = f_fft[0].real # 获取实部
        f[1] = f_fft[0].imag # 获取虚部
    return f


def add(f, g):
    """Addition of two polynomials (coefficient representation).
        多项式加法，对应每个系数进行加法，返回新的多项式
    """
    assert len(f) == len(g)
    deg = len(f)
    return [f[i] + g[i] for i in range(deg)]


def neg(f):
    """Negation of a polynomials (any representation).
        多项式取负，将多项式的每个系数取反
    """
    deg = len(f)
    return [- f[i] for i in range(deg)]


def sub(f, g):
    """Substraction of two polynomials (any representation).
        多项式减法
    """
    return add(f, neg(g)) # 减法可以通过加上负数来实现


def mul(f, g):
    """Multiplication of two polynomials (coefficient representation).
        多项式乘法，利用FFT和IFFT高效计算
    """
    return ifft(mul_fft(fft(f), fft(g))) # 乘法通过 FFT 和 IFFT 实现


def div(f, g):
    """Division of two polynomials (coefficient representation).
        多项式除法
    """
    return ifft(div_fft(fft(f), fft(g))) # 除法通过 FFT 和 IFFT 实现


def adj(f):
    """Ajoint of a polynomial (coefficient representation)."""
    return ifft(adj_fft(fft(f)))


def add_fft(f_fft, g_fft):
    """Addition of two polynomials (FFT representation)."""
    return add(f_fft, g_fft)


def sub_fft(f_fft, g_fft):
    """Substraction of two polynomials (FFT representation)."""
    return sub(f_fft, g_fft)


def mul_fft(f_fft, g_fft):
    """Multiplication of two polynomials (coefficient representation)."""
    deg = len(f_fft)
    return [f_fft[i] * g_fft[i] for i in range(deg)]


def div_fft(f_fft, g_fft):
    """Division of two polynomials (FFT representation)."""
    assert len(f_fft) == len(g_fft)
    deg = len(f_fft)
    return [f_fft[i] / g_fft[i] for i in range(deg)]


def adj_fft(f_fft):
    """Ajoint of a polynomial (FFT representation)."""
    deg = len(f_fft)
    return [f_fft[i].conjugate() for i in range(deg)]


"""This value is the ratio between:
    - The degree n
    - The number of complex coefficients of the NTT
While here this ratio is 1, it is possible to develop a short NTT such that it is 2.
"""
fft_ratio = 1
