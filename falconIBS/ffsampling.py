"""This file contains important algorithms for Falcon.

- the Fast Fourier orthogonalization (in coefficient and FFT representation)
- the Fast Fourier nearest plane (in coefficient and FFT representation)
- the Fast Fourier sampling (only in FFT)
.
"""
from falconIBS.common import split, merge                         # Split, merge
from falconIBS.fft import add, sub, mul, div, adj                 # Operations in coef.
from falconIBS.fft import add_fft, sub_fft, mul_fft, div_fft, adj_fft  # Ops in FFT
from falconIBS.fft import split_fft, merge_fft, fft_ratio         # FFT
from falconIBS.samplerz import samplerz                           # Gaussian sampler in Z


def gram(B):
    """Compute the Gram matrix of B.

    Args:
        B: a matrix

    Format: coefficient

    计算矩阵B的Gram矩阵，Garm矩阵是一个方阵，用于表示给定矩阵的列之间的内积
    """
    rows = range(len(B)) # 矩阵B的行索引
    ncols = len(B[0]) # 矩阵B的列数
    deg = len(B[0][0]) # 每个多项式的系数向量
    G = [[[0 for coef in range(deg)] for j in rows] for i in rows] # 初始化Gram矩阵
    for i in rows:
        for j in rows:
            for k in range(ncols):
                G[i][j] = add(G[i][j], mul(B[i][k], adj(B[j][k]))) # 存储的是第 i 行与第 j 行向量的内积
    return G


def ldl(G):
    """
    Compute the LDL decomposition of G. Only works with 2 * 2 matrices.

    Args:
        G: a Gram matrix

    Format: coefficient

    Corresponds to algorithm 8 (LDL*) of Falcon's documentation,
    except it's in polynomial representation.

    计算矩阵G的LDL分解，仅适用于2×2矩阵，该分解将一个对称矩阵分解为一个下三角矩阵L、一个对角矩阵D
    """
    deg = len(G[0][0])
    dim = len(G)
    assert (dim == 2)
    assert (dim == len(G[0]))

    zero = [0] * deg
    one = [1] + [0] * (deg - 1)
    D00 = G[0][0][:]
    L10 = div(G[1][0], G[0][0])
    D11 = sub(G[1][1], mul(mul(L10, adj(L10)), G[0][0]))
    L = [[one, zero], [L10, one]]
    D = [[D00, zero], [zero, D11]]

    return [L, D]


def ldl_fft(G):
    """
    Compute the LDL decomposition of G. Only works with 2 * 2 matrices.

    Args:
        G: a Gram matrix

    Format: FFT

    Corresponds to algorithm 8 (LDL*) of Falcon's documentation.

    计算G的LDL分解，仅适用于2×2矩阵，矩阵元素使用FFT表示形式，
    """
    deg = len(G[0][0])
    dim = len(G)
    assert (dim == 2)
    assert (dim == len(G[0]))

    zero = [0] * deg
    one = [1] * deg
    D00 = G[0][0][:]
    L10 = div_fft(G[1][0], G[0][0])
    D11 = sub_fft(G[1][1], mul_fft(mul_fft(L10, adj_fft(L10)), G[0][0]))
    L = [[one, zero], [L10, one]]
    D = [[D00, zero], [zero, D11]]

    return [L, D]


def ffldl(G):
    """Compute the ffLDL decomposition tree of G.

    Args:
        G: a Gram matrix

    Format: coefficient

    Corresponds to algorithm 9 (ffLDL) of Falcon's documentation,
    except it's in polynomial representation.

    计算G的ffLDL分解树，递归的分解矩阵，并生成一个包含多层次分解结果的树结构
    """
    n = len(G[0][0])
    L, D = ldl(G)
    # Coefficients of L, D are elements of R[x]/(x^n - x^(n/2) + 1), in coefficient representation
    if (n > 2):
        # A bisection is done on elements of a 2*2 diagonal matrix.
        d00, d01 = split(D[0][0])
        d10, d11 = split(D[1][1])
        G0 = [[d00, d01], [adj(d01), d00]]
        G1 = [[d10, d11], [adj(d11), d10]]
        return [L[1][0], ffldl(G0), ffldl(G1)]
    elif (n == 2):
        # Bottom of the recursion.
        D[0][0][1] = 0
        D[1][1][1] = 0
        return [L[1][0], D[0][0], D[1][1]]


def ffldl_fft(G):
    """Compute the ffLDL decomposition tree of G.

    Args:
        G: a Gram matrix

    Format: FFT

    Corresponds to algorithm 9 (ffLDL) of Falcon's documentation.

    计算 G 的 ffLDL 分解树，适用于 FFT 表示形式的输入。类似于 ffldl，但是它接受 FFT 格式的多项式
    """
    n = len(G[0][0]) * fft_ratio
    L, D = ldl_fft(G)
    # Coefficients of L, D are elements of R[x]/(x^n - x^(n/2) + 1), in FFT representation
    if (n > 2):
        # A bisection is done on elements of a 2*2 diagonal matrix.
        d00, d01 = split_fft(D[0][0])
        d10, d11 = split_fft(D[1][1])
        G0 = [[d00, d01], [adj_fft(d01), d00]]
        G1 = [[d10, d11], [adj_fft(d11), d10]]
        return [L[1][0], ffldl_fft(G0), ffldl_fft(G1)]
    elif (n == 2):
        # End of the recursion (each element is real).
        return [L[1][0], D[0][0], D[1][1]]


def ffnp(t, T):
    """Compute the ffnp reduction of t, using T as auxilary information.

    Args:
        t: a vector
        T: a ldl decomposition tree

    Format: coefficient

    计算 ffnp 还原过程，基于 T（LDL 分解树）和 t（输入向量）
    """
    n = len(t[0])
    z = [None, None]
    if (n > 1):
        l10, T0, T1 = T
        z[1] = merge(ffnp(split(t[1]), T1))
        t0b = add(t[0], mul(sub(t[1], z[1]), l10))
        z[0] = merge(ffnp(split(t0b), T0))
        return z
    elif (n == 1):
        z[0] = [round(t[0][0])]
        z[1] = [round(t[1][0])]
        return z


def ffnp_fft(t, T):
    """Compute the ffnp reduction of t, using T as auxilary information.

    Args:
        t: a vector
        T: a ldl decomposition tree

    Format: FFT

    与 ffnp 类似，但处理 FFT 格式的向量和 LDL 分解树
    """
    n = len(t[0]) * fft_ratio
    z = [0, 0]
    if (n > 1):
        l10, T0, T1 = T
        z[1] = merge_fft(ffnp_fft(split_fft(t[1]), T1))
        t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        z[0] = merge_fft(ffnp_fft(split_fft(t0b), T0))
        return z
    elif (n == 1):
        z[0] = [round(t[0][0].real)]
        z[1] = [round(t[1][0].real)]
        return z


def ffsampling_fft(t, T, sigmin, randombytes):
    """Compute the ffsampling of t, using T as auxilary information.

    Args:
        t: a vector
        T: a ldl decomposition tree

    Format: FFT

    Corresponds to algorithm 11 (ffSampling) of Falcon's documentation.

    执行 ffsampling 过程，基于 t 和 T，并使用 sigmin 和 randombytes 作为额外参数。这个过程与签名生成和验证过程相关
    """
    n = len(t[0]) * fft_ratio
    z = [0, 0]
    if (n > 1):
        l10, T0, T1 = T
        z[1] = merge_fft(ffsampling_fft(split_fft(t[1]), T1, sigmin, randombytes))
        t0b = add_fft(t[0], mul_fft(sub_fft(t[1], z[1]), l10))
        z[0] = merge_fft(ffsampling_fft(split_fft(t0b), T0, sigmin, randombytes))
        return z
    elif (n == 1):
        z[0] = [samplerz(t[0][0].real, T[0], sigmin, randombytes)]
        z[1] = [samplerz(t[1][0].real, T[0], sigmin, randombytes)]
        return z
