"""
This file implements the section 3.8.2 of Falcon's documentation.

主要处理多项式运算和NTRU相关数学计算
"""

from falconIBS.fft import fft, ifft, add_fft, mul_fft, adj_fft, div_fft
from falconIBS.fft import add, mul, div, adj
from falconIBS.ntt import ntt
from falconIBS.common import sqnorm
from falconIBS.samplerz import samplerz
from Crypto.Hash import SHAKE256
from falconIBS.rng import ChaCha20
import time
import os

q = 12 * 1024 + 1 # NTRU系统中的模数


def karatsuba(a, b, n):
    """
    Karatsuba multiplication between polynomials.
    The coefficients may be either integer or real.

    Karatsuba多项式乘法
    该函数递归地进行Karatsuba乘法，系数可以是整数或实数
    """
    if n == 1: # 如果多项式的长度为1，直接返回结果
        return [a[0] * b[0], 0]
    else:
        n2 = n // 2 # 将多项式分成两部分
        a0 = a[:n2]
        a1 = a[n2:]
        b0 = b[:n2]
        b1 = b[n2:]

        ax = [a0[i] + a1[i] for i in range(n2)] # a0和a1对应元素相加
        bx = [b0[i] + b1[i] for i in range(n2)] # b0和b1对应元素相加

        # 递归计算
        a0b0 = karatsuba(a0, b0, n2)
        a1b1 = karatsuba(a1, b1, n2)
        axbx = karatsuba(ax, bx, n2)

        # 合并结果，避免重复计算
        for i in range(n):
            axbx[i] -= (a0b0[i] + a1b1[i]) # 从 axbx 中减去 a0b0 和 a1b1 的结果
        ab = [0] * (2 * n)
        for i in range(n):
            ab[i] += a0b0[i]
            ab[i + n] += a1b1[i]
            ab[i + n2] += axbx[i]
        return ab


def karamul(a, b):
    """
    Karatsuba multiplication, followed by reduction mod (x ** n + 1).

    Karatsuba乘法，之后对结果进行模运算 mod (x ** n + 1)
    """
    n = len(a)
    ab = karatsuba(a, b, n) # 执行Karatsuba乘法
    abr = [ab[i] - ab[i + n] for i in range(n)] # 对结果进行模运算
    return abr


def galois_conjugate(a):
    """
    Galois conjugate of an element a in Q[x] / (x ** n + 1).
    Here, the Galois conjugate of a(x) is simply a(-x).

    计算元素a在Q[x] / (x ** n + 1)中的Galois共轭
    在此实现中，共轭操作是将a(x)转化为a(-x)
    """
    n = len(a)
    return [((-1) ** i) * a[i] for i in range(n)] # 返回 a(-x)，符号根据指数改变


def field_norm(a):
    """
    Project an element a of Q[x] / (x ** n + 1) onto Q[x] / (x ** (n // 2) + 1).
    Only works if n is a power-of-two.

    将元素 a 从 Q[x] / (x ** n + 1) 投影到 Q[x] / (x ** (n // 2) + 1)
    仅在 n 为 2 的幂时有效
    """
    n2 = len(a) // 2 # 分为两部分
    ae = [a[2 * i] for i in range(n2)] # 取偶数位置的项
    ao = [a[2 * i + 1] for i in range(n2)] # 取奇数位置的项
    ae_squared = karamul(ae, ae) # 对偶数项做 Karatsuba 乘法
    ao_squared = karamul(ao, ao) # 对奇数项做 Karatsuba 乘法
    res = ae_squared[:] # 初始化结果为偶数项的平方
    for i in range(n2 - 1):
        res[i + 1] -= ao_squared[i] # 处理奇数项的影响
    res[0] += ao_squared[n2 - 1] # 最后加上最后一项
    return res


def lift(a):
    """
    Lift an element a of Q[x] / (x ** (n // 2) + 1) up to Q[x] / (x ** n + 1).
    The lift of a(x) is simply a(x ** 2) seen as an element of Q[x] / (x ** n + 1).

    将元素 a 从 Q[x] / (x ** (n // 2) + 1) 提升到 Q[x] / (x ** n + 1)
    提升的方式是将 a(x) 变为 a(x ** 2)
    """
    n = len(a)
    res = [0] * (2 * n) # 结果初始化为 2n 长度
    for i in range(n):
        res[2 * i] = a[i] # 将 a(x) 的系数填充到 res 的偶数位置
    return res


def bitsize(a):
    """
    Compute the bitsize of an element of Z (not counting the sign).
    The bitsize is rounded to the next multiple of 8.
    This makes the function slightly imprecise, but faster to compute.

    计算整数 a 的 bit 大小（不包括符号位）
    结果会被四舍五入到下一个 8 的倍数
    """
    val = abs(a)
    res = 0
    while val:
        res += 8
        val >>= 8
    return res


def reduce(f, g, F, G):
    """
    Reduce (F, G) relatively to (f, g).

    This is done via Babai's reduction.59aRAMWY2b
    (F, G) <-- (F, G) - k * (f, g), where k = round((F f* + G g*) / (f f* + g g*)).
    Corresponds to algorithm 7 (Reduce) of Falcon's documentation.

    使用 Babai 的方法对 (F, G) 进行约简，相对于 (f, g)
    算法 7 (Reduce) 中的步骤
    """
    n = len(f)
    size = max(53, bitsize(min(f)), bitsize(max(f)), bitsize(min(g)), bitsize(max(g)))

    f_adjust = [elt >> (size - 53) for elt in f] # 调整f的精度
    g_adjust = [elt >> (size - 53) for elt in g] # 调整g的精度
    fa_fft = fft(f_adjust) # 对f进行FFT
    ga_fft = fft(g_adjust) # 对g进行FFT

    while(1):
        # Because we work in finite precision to reduce very large polynomials,
        # we may need to perform the reduction several times.
        Size = max(53, bitsize(min(F)), bitsize(max(F)), bitsize(min(G)), bitsize(max(G)))
        if Size < size:
            break

        F_adjust = [elt >> (Size - 53) for elt in F]
        G_adjust = [elt >> (Size - 53) for elt in G]
        Fa_fft = fft(F_adjust)
        Ga_fft = fft(G_adjust)

        den_fft = add_fft(mul_fft(fa_fft, adj_fft(fa_fft)), mul_fft(ga_fft, adj_fft(ga_fft)))
        num_fft = add_fft(mul_fft(Fa_fft, adj_fft(fa_fft)), mul_fft(Ga_fft, adj_fft(ga_fft)))
        k_fft = div_fft(num_fft, den_fft)
        k = ifft(k_fft)
        k = [int(round(elt)) for elt in k]
        if all(elt == 0 for elt in k): # 如果k全为0，退出
            break
        # The two next lines are the costliest operations in ntru_gen
        # (more than 75% of the total cost in dimension n = 1024).
        # There are at least two ways to make them faster:
        # - replace Karatsuba with Toom-Cook
        # - mutualized Karatsuba, see ia.cr/2020/268
        # For simplicity reasons, we didn't implement these optimisations here.
        fk = karamul(f, k)
        gk = karamul(g, k)
        for i in range(n):
            F[i] -= fk[i] << (Size - size)
            G[i] -= gk[i] << (Size - size)
    return F, G


def xgcd(b, n):
    """
    Compute the extended GCD of two integers b and n.
    Return d, u, v such that d = u * b + v * n, and d is the GCD of b, n.
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while n != 0:
        q, b, n = b // n, n, b % n
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


def ntru_solve(f, g):
    """
    Solve the NTRU equation for f and g.
    Corresponds to NTRUSolve in Falcon's documentation.
    """
    n = len(f)
    if n == 1:
        f0 = f[0]
        g0 = g[0]
        d, u, v = xgcd(f0, g0)
        if d != 1:
            raise ValueError
        else:
            return [- q * v], [q * u]
    else:
        fp = field_norm(f)
        gp = field_norm(g)
        Fp, Gp = ntru_solve(fp, gp)
        F = karamul(lift(Fp), galois_conjugate(g))
        G = karamul(lift(Gp), galois_conjugate(f))
        F, G = reduce(f, g, F, G)
        return F, G


def gs_norm(f, g, q):
    """
    Compute the squared Gram-Schmidt norm of the NTRU matrix generated by f, g.
    This matrix is [[g, - f], [G, - F]].
    This algorithm is equivalent to line 9 of algorithm 5 (NTRUGen).
    """
    sqnorm_fg = sqnorm([f, g])
    ffgg = add(mul(f, adj(f)), mul(g, adj(g)))
    Ft = div(adj(g), ffgg)
    Gt = div(adj(f), ffgg)
    sqnorm_FG = (q ** 2) * sqnorm([Ft, Gt])
    return max(sqnorm_fg, sqnorm_FG)


def gen_poly(n):
    """
    Generate a polynomial of degree at most (n - 1), with coefficients
    following a discrete Gaussian distribution D_{Z, 0, sigma_fg} with
    sigma_fg = 1.17 * sqrt(q / (2 * n)).
    """
    # 1.17 * sqrt(12289 / 8192)
    sigma = 1.43300980528773
    assert(n < 4096)
    f0 = [samplerz(0, sigma, sigma - 0.001) for _ in range(4096)]
    f = [0] * n
    k = 4096 // n
    for i in range(n):
        # We use the fact that adding k Gaussian samples of std. dev. sigma
        # gives a Gaussian sample of std. dev. sqrt(k) * sigma.
        f[i] = sum(f0[i * k + j] for j in range(k))
    return f

# 使用用户ID参与生成多项式
def gen_poly_with_id(n, identity, sigma=1.43300980528773):
    """
    根据用户ID生成多项式，系数遵循离散高斯分布。

    参数:
        n: 多项式维度
        identity: 用户身份字符串
        sigma: 高斯分布标准差，默认值1.43300980528773
    """
    start_gen_poly = time.perf_counter()
    # 使用SHAKE256生成基于ID的种子
    shake = SHAKE256.new()
    shake.update(identity.encode())

    shake.update(os.urandom(16)) # 增加随机熵

    seed = shake.read(32)  # 生成32字节种子

    # 初始化ChaCha20 PRG
    prng = ChaCha20(seed)

    endPRG = time.perf_counter()
    # print(f"PRG生成时间：{endPRG-start_gen_poly:.4f}s")

    # 生成多项式系数
    f0 = [samplerz(0, sigma, sigma - 0.001, prng.randombytes) for _ in range(4096)]
    f = [0] * n
    k = 4096 // n
    for i in range(n):
        f[i] = sum(f0[i * k + j] for j in range(k))
    endgen_poly = time.perf_counter()
    # print(f"多项式生成时间：{endgen_poly-start_gen_poly:.4f}秒")
    return f



def ntru_gen(n, identity=None):
    """
    Implement the algorithm 5 (NTRUGen) of Falcon's documentation.
    At the end of the function, polynomials f, g, F, G in Z[x]/(x ** n + 1)
    are output, which verify f * G - g * F = q mod (x ** n + 1).
    """

    total_start = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        iter_start = time.perf_counter()
        if identity is None:
            # 生成随机主密钥（PKG用）
            f = gen_poly(n)
            g = gen_poly(n)
        else:
            # 生成与ID绑定的用户密钥
            f = gen_poly_with_id(n, identity)
            g = gen_poly_with_id(n, identity + "_g")  # 避免f和g相同

        # 检查Gram-Schmidt范数
        gs_start = time.perf_counter()
        if gs_norm(f, g, q) > (1.17 ** 2) * q:
            print(f"尝试 {attempts} - gs_norm 检查失败: {time.perf_counter() - gs_start:.4f} 秒 --"
                  f"--(值: {gs_norm(f, g, q):.2f}, 阈值: {(1.17 ** 2) * q:.2f})")
            continue

        # 检查f在NTT域中是否可逆
        ntt_start = time.perf_counter()
        f_ntt = ntt(f)
        if any((elem == 0) for elem in f_ntt):
            print(f"尝试 {attempts} - NTT 可逆性检查失败: {time.perf_counter() - ntt_start:.4f} 秒")
            continue

        solve_start = time.perf_counter()
        try:
            F, G = ntru_solve(f, g)
            F = [int(coef) for coef in F]
            G = [int(coef) for coef in G]
            # print(f"ntru_solve 运行时间: {time.perf_counter() - solve_start:.4f} 秒")

            total_end = time.perf_counter()
            # print(f"总运行时间: {total_end - total_start:.4f} 秒, 尝试次数: {attempts}")
            return f, g, F, G
        # If the NTRU equation cannot be solved, a ValueError is raised
        # In this case, we start again
        except ValueError:
            print(f"尝试 {attempts} - ntru_solve 失败: {time.perf_counter() - solve_start:.4f} 秒")
            continue


# if __name__ == "__main__":
#     # 生成PKG主密钥
#     f_m, g_m, F_m, G_m = ntru_gen(512)
#     print("f_m:",f_m)
#     print("g_m:",g_m)
#     print("F_m:",F_m)
#     print("G_m:",G_m)
#     print("PKG主密钥生成完成")
#
#     # 生成用户密钥
#     identity = "1"
#     f_u, g_u, F_u, G_u = ntru_gen(512, identity)
#     print("f_u:", f_u)
#     print("g_u:", g_u)
#     print("F_u:", F_u)
#     print("G_u:", G_u)
#     print("用户密钥生成完成")


