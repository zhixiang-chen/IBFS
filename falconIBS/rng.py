"""
Implementation of the RNG used during the signing procedure.
This RNG is based on ChaCha20. The 56-bytes seed is split into
14 words s[0], ..., s[13] of 32 bits each. s[12], s[13] define
a 64-bit counter ctr = s[12] + s[13] << 32

Random bits are generated as follow:
- fill the ChaCha20 matrix as follows:
    CW[0]  CW[1]  CW[2]  CW[3]
     s[0]   s[1]   s[2]   s[3]
     s[4]   s[5]   s[6]   s[7]
     s[8]   s[9]   s[1]   s[1]
- generate 512 bits of randomness by applying the block function as
  in "regular" ChaCha20 (e.g. https://tools.ietf.org/html/rfc7539)
- increment ctr
For efficiency reasons, the reference code generates 8 chunks of randomness
at a time (hence 512 * 8 = 4096 bits), and interleave the outputs by blocks
of 32 bits. For reproducibility, we do the same here.

实现了基于ChaCha20 的伪随机数生成器（PRG），用于签名过程中生成随机数
"""

# ChaCha20 constants
CW = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]


def roll(x, n):
    """
    The roll function
    Lifted from https://www.johndcook.com/blog/2019/03/03/do-the-chacha/

    给定给定一个 32 位整数 x 和一个位移量 n，roll 函数通过将 x 左移 n 位并将溢出部分右移，从而对 x 执行循环移位
    & 0xffffffff 确保结果是一个 32 位整数
    """
    return ((x << n) & 0xffffffff) + (x >> (32 - n))


class ChaCha20:

    def __init__(self, src):
        """
        Initialize the PRG. src is the initial seed, ctr is the counter,
        and hexbytes is a buffer for the pseudorandom output.

        初始化PRG
        传入一个 56 字节的初始种子 src，并将其分割成 14 个 32 位的整数（存储在 self.s 中）来初始化
        self.ctr 是一个 64 位的计数器，使用 s[12] 和 s[13] 来表示
        hexbytes 是一个缓冲区，用于存储生成的伪随机输出
        """
        self.s = [int.from_bytes(src[4 * i: 4 * (i + 1)], "little") for i in range(14)]
        self.ctr = self.s[12] + (self.s[13] << 32)
        self.hexbytes = ""

    def __repr__(self):
        """
        Print the PRG state.

        打印PRG的当前状态
        """
        rep = "s = ["
        for elt in self.s:
            rep += '0x{:08x}, '.format(elt)
        rep = rep[:-2] + "]\n"
        rep += "ctr = " + str(self.ctr)
        return rep

    def qround(self, A, B, C, D):
        """
        Quarter-round function.
        Lifted from https://www.johndcook.com/blog/2019/03/03/do-the-chacha/,
        then modified.

        qround 方法是 ChaCha20 算法中的核心操作之一，执行所谓的“quarter-round”函数
        它对四个状态变量（A、B、C、D）执行加法、异或和旋转操作
        每次调用 qround，这些值会被修改并更新到 self.state 中
        """
        a = self.state[A]
        b = self.state[B]
        c = self.state[C]
        d = self.state[D]
        a = (a + b) & 0xffffffff
        d = roll(d ^ a, 16)
        c = (c + d) & 0xffffffff
        b = roll(b ^ c, 12)
        a = (a + b) & 0xffffffff
        d = roll(d ^ a, 8)
        c = (c + d) & 0xffffffff
        b = roll(b ^ c, 7)
        self.state[A] = a
        self.state[B] = b
        self.state[C] = c
        self.state[D] = d

    def update(self):
        """
        One update of the ChaCha20 PRG.

        update 方法是 ChaCha20 算法的一次完整更新操作
        它首先初始化 ChaCha20 的状态矩阵 state，并将种子和计数器值填充到 state 中
        然后，执行 10 次 qround 来进行状态更新。最后，更新计数器 ctr
        """
        self.state = [0] * 16
        self.state[0:4] = CW[:]
        self.state[4:14] = [self.s[i] for i in range(10)]
        self.state[14] = self.s[10] ^ (self.ctr & 0xffffffff)
        self.state[15] = self.s[11] ^ (self.ctr >> 32)
        state = self.state[:]
        for _ in range(10):
            self.qround(0, 4, 8, 12)
            self.qround(1, 5, 9, 13)
            self.qround(2, 6, 10, 14)
            self.qround(3, 7, 11, 15)
            self.qround(0, 5, 10, 15)
            self.qround(1, 6, 11, 12)
            self.qround(2, 7, 8, 13)
            self.qround(3, 4, 9, 14)
        for i in range(16):
            self.state[i] = (self.state[i] + state[i]) & 0xffffffff
        self.ctr += 1
        return self.state

    def block_update(self):
        """
        Produces 8 consecurite updates, and interleave the results.

        block_update 方法通过调用 update 方法 8 次来生成 4096 位的伪随机数据，并将这些更新结果交错在一起
        每次调用 update 后，会将结果转化为 4 字节的小端字节序，并转换为 16 进制格式
        最终返回一个由这些字节组成的字符串
        """
        block = [None] * 16 * 8
        for i in range(8):
            block[i::8] = self.update()
        return "".join(elt.to_bytes(4, "little").hex() for elt in block)

    def randombytes(self, k):
        """
        Generate random bytes.
        Perform some shenanigans to match the reference code PRG.

        randombytes 方法用于生成 k 字节的随机数
        如果当前缓冲区中的伪随机字节不够，它会调用 block_update 来生成更多的字节
        然后，它从缓冲区中提取所需数量的字节，转换为字节数组，并返回
        返回的字节是反转顺序的，确保与参考代码保持一致
        """
        if (2 * k > len(self.hexbytes)):
            self.hexbytes = self.block_update()
        out = self.hexbytes[:(2 * k)]
        out = "".join(out[i:i + 2] for i in range(2 * k - 2, -1, -2))
        self.hexbytes = self.hexbytes[(2 * k):]
        return bytes.fromhex(out)[::-1]
