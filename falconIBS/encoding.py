"""
Compression and decompression routines for signatures.
压缩和解压缩签名
函数：compress(v, slen)将整数列表压缩成一个字节串，使用符号、低位和一元编码来表示每个整数的值
     decompress(x, slen, n)将字节串解压成原始的整数列表，恢复符号、低位和高位值，并处理一些特殊情况（如无效编码）
"""


def compress(v, slen):
    """
    Take as input a list of integers v and a bytelength slen, and
    return a bytestring of length slen that encode/compress v.
    If this is not possible, return False.

    For each coefficient of v:
    - the sign is encoded on 1 bit
    - the 7 lower bits are encoded naively (binary)
    - the high bits are encoded in unary encoding

    该函数用于将一个整数列表 v 压缩成一个字节串（bytes），字节串的长度为 slen。如果压缩后的数据过长，返回 False
    """
    u = "" # 初始化一个空字符串 u，用于存储二进制编码后的数据
    for coef in v: # 遍历整数列表 v 中的每个系数
        # Encode the sign
        s = "1" if coef < 0 else "0" # 如果系数为负，使用 '1'，否则使用 '0'
        # Encode the low bits
        s += format((abs(coef) % (1 << 7)), '#09b')[2:] # 获取系数的低 7 位，并转换为二进制字符串
        # Encode the high bits
        s += "0" * (abs(coef) >> 7) + "1" # 高位使用一元编码，0 的个数表示高位的大小，最后加上 1 表示结束
        u += s # 将编码结果添加到 u 字符串中
    # The encoding is too long
    # 如果编码的结果长度超过了目标字节长度，返回 False
    if len(u) > 8 * slen:
        return False
    # 如果编码长度小于目标长度，使用 '0' 填充至目标长度
    u += "0" * (8 * slen - len(u))

    # 将二进制字符串 u 分割成每 8 位一个字节，并转换成字节列表
    w = [int(u[8 * i: 8 * i + 8], 2) for i in range(len(u) // 8)]

    # 将字节列表转换为字节串并返回
    x = bytes(w)
    return x


def decompress(x, slen, n):
    """
    Take as input an encoding x, a bytelength slen and a length n, and
    return a list of integers v of length n such that x encode v.
    If such a list does not exist, the encoding is invalid and we output False.

    该函数用于将一个压缩后的字节串 x 解压为一个整数列表 v，该列表的长度为 n
    """
    if (len(x) > slen): # 如果输入字节串 x 的长度超过了预定长度 slen，返回 False
        print("Too long")
        return False
    w = list(x) # 将字节串 x 转换为字节列表 w
    u = ""

    # 将字节列表转换为二进制字符串 u（将每个字节转换为 8 位二进制表示，并拼接起来）
    for elt in w:
        u += bin((1 << 8) ^ elt)[3:]
    v = [] # 用于存储解压后的整数列表

    # Remove the last bits
    # 移除最后填充的零
    while u[-1] == "0":
        u = u[:-1]

    try:
        # 依次从二进制字符串中恢复整数
        while (u != "") and (len(v) < n):
            # Recover the sign of coef
            sign = -1 if u[0] == "1" else 1 # 恢复符号位
            # Recover the 7 low bits of abs(coef)
            low = int(u[1:8], 2) # 恢复低 7 位的值
            i, high = 8, 0 # 初始化高位
            # Recover the high bits of abs(coef)
            while (u[i] == "0"): # 高位部分通过一元编码恢复
                i += 1
                high += 1
            # Compute coef
            # 计算原始系数
            coef = sign * (low + (high << 7))
            # Enforce a unique encoding for coef = 0
            # 如果系数为 0 且符号为负，返回 False，因为这种情况在编码中是无效的
            if (coef == 0) and (sign == -1):
                return False
            # Store intermediate results
            # 将解压出来的系数添加到列表 v 中
            v += [coef]
            u = u[i + 1:] # 更新 u，去掉已经处理过的部分
        # In this case, the encoding is invalid
        # 如果解压后的系数数量不等于 n，返回 False
        if (len(v) != n):
            return False
        return v
    # IndexError is raised if indices are read outside the table bounds
    except IndexError:
        return False # 如果读取超出字符串范围，返回 False
