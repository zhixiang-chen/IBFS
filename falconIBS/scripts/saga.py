"""
This is a light version of SAGA: https://github.com/PQShield/SAGA
"""
# Estimators for moments
from scipy.stats import skew, kurtosis, moment
# Statistical (normality) tests
from scipy.stats import chisquare
# Distributions
from scipy.stats import chi2
# Numpy stuff
from numpy import cov, set_printoptions, diag, array, mean
from numpy.linalg import matrix_rank, inv, eigh
import matplotlib.pyplot as plt

# Math functions
from math import ceil, sqrt, exp, log
# Data management
from copy import deepcopy
import re
import pandas

# For HZ multivariate test, used in scipy.spatial.distance.mahalanobis
from numpy import floor
from numpy import tile

# qqplot
import scipy.stats as stats
from numpy import transpose, sort

# doornik hansen
from numpy import corrcoef, power
from numpy import log as nplog
from numpy import sqrt as npsqrt

# For debugging purposes
import time

# Tailcut rate
# 尾切率，值的尾端被切除的比例和阈值，优化和过滤异常数据
tau = 14
# Minimal size of a bucket for the chi-squared test (must be >= 5)
# 卡方检验的参数
chi2_bucket = 10
# Minimal p-value
# 最小p值
pmin = 0.001
# Print options
# 设置数字打印时的精度
set_printoptions(precision=4)


def gaussian(x, mu, sigma):
    """
    Gaussian function of center mu and "standard deviation" sigma.
    输入：
        x：自变量，表示高斯函数的输入值
        mu：高斯分布的均值，即分布中心
        sigma：高斯分布的标准差，控制分布的宽度
    输出：
        返回x对应的高斯分布值
    """
    return exp(- ((x - mu) ** 2) / (2 * (sigma ** 2)))


def make_gaussian_pdt(mu, sigma):
    """
    Make the probability distribution table (PDT) of a discrete Gaussian.
    The output is a dictionary.

    离散高斯分布的概率分布
    输入：
        mu：高斯分布的均值
        sigma：高斯分布的标准差
    输出：
        返回一个字典（pdt），其中键是整数 z，值是相应的高斯分布值,每个值表示对应的 z 在该高斯分布中的概率

    """
    # The distribution is restricted to [-zmax, zmax).
    zmax = int(ceil(tau * sigma)) # 计算高斯分布的最大偏离值
    pdt = dict() # 创建一个空字典，用于存储离散化的高斯概率分布
    for z in range(int(floor(mu)) - zmax, int(ceil(mu)) + zmax):
        pdt[z] = gaussian(z, mu, sigma)
    gauss_sum = sum(pdt.values()) # 字典中所有值的总和，用于归一化
    for z in pdt:
        pdt[z] /= gauss_sum # 遍历字典中的每一个值，进行归一化
    return pdt


class UnivariateSamples:
    """
    Class for computing statistics on univariate Gaussian samples.
    计算单变量高斯样本的统计信息
    比较离散高斯分布的预期分布（理论分布）和实际样本的经验分布
    进行一些统计分析，如均值、标准差、偏度、峰度以及卡方检验
    """

    def __init__(self, mu, sigma, list_samples):
        """
        Input:
        - the expected center mu of a discrete Gaussian over Z
        - the expected standard deviation sigma of a discrete Gaussian over Z
        - a list of samples defining an empiric distribution

        Output:
        - the means of the expected and empiric distributions
        - the standard deviations of the expected and empiric distributions
        - the skewness of the expected and empiric distributions
        - the kurtosis of the expected and empiric distributions
        - a chi-square test between the two distributions

        输入：
            mu:离散高斯分布的理论均值
            sigma:离散高斯分布的理论标准差
            list_samples:样本列表，定义了经验分布
        输出:
            计算并输出理论分布和经验分布的均值、标准差、偏度、峰度
            进行卡方检验，比较两个分布的差异
        """
        zmax = int(ceil(tau * sigma)) # 计算离散化高斯分布的最大偏差值
        # Expected center standard variation.
        self.exp_mu = mu
        self.exp_sigma = sigma
        # Number of samples
        self.nsamples = len(list_samples) # 记录样本数
        self.histogram = dict() # 存储样本的频率分布
        self.outlier = 0 # 记录异常值的数量
        # Initialize histogram
        # 设置离散化的范围，并初始化频率分布
        start = int(floor(mu)) - zmax
        end = int(ceil(mu)) + zmax
        for z in range(start, end):
            self.histogram[z] = 0
        for z in list_samples: # 遍历样本list_samples，如果样本不在频率分布范围内，认为是异常值，计数outlier
            # Detect and count outliers (samples not in [-zmax, zmax))
            if z not in self.histogram:
                self.outlier += 1
            # Fill histogram according to the samples
            else:
                self.histogram[z] += 1
        # Empiric mean, variance, skewness, kurtosis and standard deviation
        # 计算经验分布的均值、方差、偏度、峰度和标准差
        self.mean = sum(list_samples) / self.nsamples
        self.variance = moment(list_samples, 2)
        self.skewness = skew(list_samples)
        self.kurtosis = kurtosis(list_samples)
        self.stdev = sqrt(self.variance)
        # Chi-square statistic and p-value
        # 进行卡方检验，比较经验分布和预期分布的差异
        self.chi2_stat, self.chi2_pvalue = self.chisquare()
        # Final assessment: the dataset is valid if:
        # - the chi-square p-value is higher than pmin
        # - there is no outlier
        self.is_valid = True
        self.is_valid &= (self.chi2_pvalue > pmin)
        self.is_valid &= (self.outlier == 0)


    def __repr__(self):
        """
        Print the sample statistics in a readable form.
        打印样本统计信息的可读形式
        """
        rep = "\n"
        rep += "Testing a Gaussian sampler with center = {c} and sigma = {s}\n".format(c=self.exp_mu, s=self.exp_sigma)
        rep += "Number of samples: {nsamples}\n\n".format(nsamples=self.nsamples)
        rep += "Moments  |   Expected     Empiric\n"
        rep += "---------+----------------------\n"
        rep += "Mean:    |   {exp:.5f}      {emp:.5f}\n".format(exp=self.exp_mu, emp=self.mean)
        rep += "St. dev. |   {exp:.5f}      {emp:.5f}\n".format(exp=self.exp_sigma, emp=self.stdev)
        rep += "Skewness |   {exp:.5f}      {emp:.5f}\n".format(exp=0, emp=self.skewness)
        rep += "Kurtosis |   {exp:.5f}      {emp:.5f}\n".format(exp=0, emp=self.kurtosis)
        rep += "\n"
        rep += "Chi-2 statistic:   {stat}\n".format(stat=self.chi2_stat)
        rep += "Chi-2 p-value:     {pval}   (should be > {p})\n".format(pval=self.chi2_pvalue, p=pmin)
        rep += "\n"
        rep += "How many outliers? {o}".format(o=self.outlier)
        rep += "\n\n"
        rep += "Is the sample valid? {i}".format(i=self.is_valid)
        return rep

    def chisquare(self):
        """
        Run a chi-square test to compare the expected and empiric distributions
        执行卡方检验，比较预期分布和经验分布之间的差异
        """
        # We construct two histograms:
        # - the expected one (exp_histogram)
        # - the empirical one (histogram)
        histogram = deepcopy(self.histogram) # 深拷贝
        # The chi-square test require buckets to have enough elements,
        # so we aggregate samples in the left and right tails in two buckets
        # 生成预期高斯分布
        exp_histogram = make_gaussian_pdt(self.exp_mu, self.exp_sigma)
        # 提取实际分布和预期分布的概率值
        obs = list(histogram.values())
        exp = list(exp_histogram.values())
        z = 0
        while(1):
            if (z >= len(exp) - 1):
                break
            while (z < len(exp) - 1) and (exp[z] < chi2_bucket / self.nsamples):
                obs[z + 1] += obs[z]
                exp[z + 1] += exp[z]
                obs.pop(z)
                exp.pop(z)
            z += 1
        obs[-2] += obs[-1]
        exp[-2] += exp[-1]
        obs.pop(-1)
        exp.pop(-1)
        exp = [round(prob * self.nsamples) for prob in exp]
        diff = self.nsamples - sum(exp_histogram.values())
        exp_histogram[int(round(self.exp_mu))] += diff
        res = chisquare(obs, f_exp=exp)
        return res


class MultivariateSamples:
    """
    Class for computing statistics on multivariate Gaussian samples
    多变量高斯分布进行统计分析
    """

    def __init__(self, sigma, list_samples):
        """
        Input:
        - sigma: an expected standard deviation
        - list_samples: a list of (expected) multivariate samples

        Output:
        - univariates[]: a list of UnivariateSamples objects (one / coordinate)
        - covariance: an empiric covariance matrix
        - DH, AS, PO, PA: statistics and p-values for the Doornik-Hansen test
        - dc_pvalue: a p-value for our custom covariance-based test

        输入:
            sigma:预期标准差
            list_samples:包含多变量样本的列表
        输出:
            univariates[]:每个维度的UnivariateSamples对象
            covariance:协方差矩阵
            DH, AS, PO, PA:Doornik-Hansen多元正态性检验的统计数据和p值
            dc_pvalue:自定义的基于协方差的检验p值
        """
        # Parse the signatures and store them
        self.nsamples = len(list_samples) # 样本数
        self.dim = len(list_samples[0]) # 样本维度
        self.data = pandas.DataFrame(list_samples)
        # Expected center and standard deviation
        self.exp_mu = 0
        self.exp_si = sigma
        # Testing sphericity
        # For each coordinate, perform an univariate analysis
        self.univariates = [None] * self.dim
        for i in range(self.dim):
            self.univariates[i] = UnivariateSamples(0, sigma, self.data[i])
        self.nb_gaussian_coord = sum((self.univariates[i].chi2_pvalue > pmin) for i in range(self.dim))
        # Estimate the (normalized) covariance matrix
        self.covariance = cov(self.data.transpose()) / (self.exp_si ** 2)
        self.DH, self.AS, self.PO, self.PA = doornik_hansen(self.data)
        self.dc_pvalue = diagcov(self.covariance, self.nsamples)

    def __repr__(self):
        """
        Print the sample statistics in a readable form.
        """
        rep = "\n"
        rep += "Testing a centered multivariate Gaussian of dimension = {dim} and sigma = {s:.3f}\n".format(dim=self.dim, s=self.exp_si)
        rep += "Number of samples: {nsamples}\n".format(nsamples=self.nsamples)
        rep += "\n"
        rep += "The test checks that the data corresponds to a multivariate Gaussian, by doing the following:\n"
        rep += "1 - Print the covariance matrix (visual check). One can also plot\n"
        rep += "    the covariance matrix by using self.show_covariance()).\n"
        rep += "2 - Perform the Doornik-Hansen test of multivariate normality.\n"
        rep += "    The p-value obtained should be > {p}\n".format(p=pmin)
        rep += "3 - Perform a custom test called covariance diagonals test.\n"
        rep += "4 - Run a test of univariate normality on each coordinate\n"
        rep += "\n"
        rep += "1 - Covariance matrix ({dim} x {dim}):\n{cov}\n".format(dim=self.dim, cov=self.covariance)
        rep += "\n"
        if (self.nsamples < 4 * self.dim):
            rep += "Warning: it is advised to have at least 8 times more samples than the dimension n.\n"
        rep += "2 - P-value of Doornik-Hansen test:                {p:.4f}\n".format(p=self.PO)
        rep += "\n"
        rep += "3 - P-value of covariance diagonals test:          {p:.4f}\n".format(p=self.dc_pvalue)
        rep += "\n"
        rep += "4 - Gaussian coordinates (w/ st. dev. = sigma)?    {k} out of {dim}\n".format(k=self.nb_gaussian_coord, dim=self.dim)
        return rep

    def show_covariance(self):
        """
        Visual representation of the covariance matrix
        """
        plt.imshow(self.covariance, interpolation='nearest')
        plt.show()


def doornik_hansen(data):
    """
    Perform the Doornik-Hansen test
    (https://doi.org/10.1111/j.1468-0084.2008.00537.x)

    This computes and transforms multivariate variants of the skewness
    and kurtosis, then computes a chi-square statistic on the results.

    函数实现了 Doornik-Hansen 正态性检验，它主要用于检验数据集是否符合多变量正态分布
    通过计算偏度（Skewness）和峰度（Kurtosis）的变种，并用卡方分布对结果进行统计检验

    执行Doornik-Hansen正态性检验
    计算并转换偏度和峰度的多变量变种，然后对结果计算卡方统计量
    """
    data = pandas.DataFrame(data)
    data = deepcopy(data)

    n = len(data) # 样本数量
    p = len(data.columns) # 数据维度
    # R is the correlation matrix, a scaling of the covariance matrix
    # R has dimensions dim * dim
    R = corrcoef(data.transpose()) # 计算数据的相关矩阵R，协方差矩阵的标准化形式
    L, V = eigh(R) # 对相关矩阵R进行特征值分解
    for i in range(p):
        if(L[i] <= 1e-12): # 如果特征值非常小，则将其设为0
            L[i] = 0
        if(L[i] > 1e-12): # 如果特征值较大，则取其倒数
            L[i] = 1 / sqrt(L[i])
    L = diag(L) # 将特征值L转化为对角矩阵

    # 如果相关矩阵的秩小于p（矩阵不可逆），需要减少维度
    if(matrix_rank(R) < p):
        V = pandas.DataFrame(V)
        G = V.loc[:, (L != 0).any(axis=0)] # 只保留特征值非零的特征向量
        data = data.dot(G) # 用特征向量矩阵G对数据进行变换
        ppre = p # 保存初始维度
        p = data.size / len(data) # 更新维度
        raise ValueError("NOTE:Due that some eigenvalue resulted zero, \
                          a new data matrix was created. Initial number \
                          of variables = ",
                         ppre, ", were reduced to = ", p)
        R = corrcoef(data.transpose()) # 重新计算相关矩阵
        L, V = eigh(R) # 再次对新矩阵进行特征值分解
        L = diag(L) # 更新特征值矩阵

    # 计算样本均值和标准差
    means = [list(data.mean())] * n
    stddev = [list(data.std(ddof=0))] * n

    # 标准化数据：Z = (数据 - 均值) / 标准差
    Z = (data - pandas.DataFrame(means)) / pandas.DataFrame(stddev)
    # Zp和Zpp是通过数据矩阵变换得到的中间矩阵
    Zp = Z.dot(V) # Zp是Z与V的乘积
    Zpp = Zp.dot(L) # Zpp是Zp与L的乘积
    # st是进一步变换的结果
    st = Zpp.dot(transpose(V))

    # skew is the multivariate skewness (dimension dim)
    # kurt is the multivariate kurtosis (dimension dim)
    # 偏度：计算多变量数据的偏度
    skew = mean(power(st, 3), axis=0)
    # 峰度：计算多变量数据的峰度
    kurt = mean(power(st, 4), axis=0)

    # Transform the skewness into a standard normal z1
    # 将偏度转换为标准正态分布的z1值
    n2 = n * n
    b = 3 * (n2 + 27 * n - 70) * (n + 1) * (n + 3)
    b /= (n - 2) * (n + 5) * (n + 7) * (n + 9)
    w2 = -1 + sqrt(2 * (b - 1))
    d = 1 / sqrt(log(sqrt(w2)))
    y = skew * sqrt((w2 - 1) * (n + 1) * (n + 3) / (12 * (n - 2)))
    # Use numpy log/sqrt as math versions dont have array input
    z1 = d * nplog(y + npsqrt(y * y + 1))

    # Transform the kurtosis into a standard normal z2
    # 将峰度转换为标准正态分布的z2值
    d = (n - 3) * (n + 1) * (n2 + 15 * n - 4)
    a = (n - 2) * (n + 5) * (n + 7) * (n2 + 27 * n - 70) / (6 * d)
    c = (n - 7) * (n + 5) * (n + 7) * (n2 + 2 * n - 5) / (6 * d)
    k = (n + 5) * (n + 7) * (n * n2 + 37 * n2 + 11 * n - 313) / (12 * d)
    al = a + (skew ** 2) * c
    chi = (kurt - 1 - (skew ** 2)) * k * 2
    z2 = (((chi / (2 * al)) ** (1 / 3)) - 1 + 1 / (9 * al)) * npsqrt(9 * al)
    kurt -= 3

    # omnibus normality statistic
    # 计算Doornik-Hansen综合正态性统计量
    DH = z1.dot(z1.transpose()) + z2.dot(z2.transpose())
    # 计算偏度和峰度的合并统计量
    AS = n / 6 * skew.dot(skew.transpose()) + n / 24 * kurt.dot(kurt.transpose())
    # degrees of freedom
    v = 2 * p
    # p-values
    PO = 1 - chi2.cdf(DH, v) # Doornik-Hansen检验的p值
    PA = 1 - chi2.cdf(AS, v) # 自定义的偏度峰度检验的p值

    return DH, AS, PO, PA


def diagcov(cov_mat, nsamples):
    """
    函数用于计算协方差矩阵的标准化对角线和的卡方检验。
    它的目的是检验协方差矩阵的对角线元素是否符合正态分布的假设
    This test studies the population covariance matrix.
    Suppose it is of this form:
     ____________
    |     |     |
    |  1  |  3  |
    |_____|_____|
    |     |     |
    |     |  2  |
    |_____|_____|

    The test will first compute sums of elements on diagonals of 1, 2 or 3,
    and store them in the table diagsum of size 2 * dim:
    - First (dim / 2) lines = means of each diag. of 1 above leading diag.
    - Following (dim / 2) lines = means of each diag. of 2 above leading diag.
    - Following (dim / 2) lines = means of each diag. of 3 above leading diag.
    - Last (dim / 2) lines = means of each diag. of 3 below leading diag.

    We are making the assumption that each cell of the covariance matrix
    follows a normal distribution of variance 1 / n. Assuming independence
    of each cell in a diagonal, each diagonal sum of k elements should
    follow a normal distribution of variance k / n (hence of variance
    1 after normalization by n / k).

    We then compute the sum of the squares of all elements in diagnorm.
    If is supposed to look like a chi-square distribution

    该测试首先计算矩阵对角线上的元素和，分为1、2或3个组，并将它们存储在大小为2 * dim的表格diagsum中：
    - 前(dim / 2)行是每个1对角线的平均值；
    - 后(dim / 2)行是每个2对角线的平均值；
    - 再后(dim / 2)行是每个3对角线的平均值；
    - 最后(dim / 2)行是每个3对角线下对角线的平均值。

    假设协方差矩阵的每个元素都服从均值为0、方差为1/n的正态分布。
    假设每个对角线的独立性，k个元素的对角线和应该服从方差为k/n的正态分布（因此，经过n/k标准化后，方差为1）。

    然后计算诊断标准化值的平方和。假设它应该服从卡方分布。
    """
    dim = len(cov_mat) # 协方差矩阵的维度
    n0 = dim // 2 # 一半的维度，控制对角线划分
    diagsum = [0] * (2 * dim) # 存储各对角线元素和的列表

    # 计算上对角线和下对角线的元素和
    for i in range(1, n0):
        diagsum[i] = sum(cov_mat[j][i + j] for j in range(n0 - i)) # 第一组对角线
        diagsum[i + n0] = sum(cov_mat[n0 + j][n0 + i + j] for j in range(n0 - i)) # 第二组对角线
        diagsum[i + 2 * n0] = sum(cov_mat[j][n0 + i + j] for j in range(n0 - i)) # 第三组对角线
        diagsum[i + 3 * n0] = sum(cov_mat[j][n0 - i + j] for j in range(n0 - i)) # 第四组对角线
    # Diagnorm contains the normalized sums, which should be normal
    # diagnorm包含标准化后的对角线元素和，这些和应该服从正态分布
    diagnorm = diagsum[:]
    for i in range(1, n0):
        nfactor = sqrt(nsamples / (n0 - i)) # 标准化因子
        diagnorm[i] *= nfactor # 计算标准化的对角线元素和
        diagnorm[i + n0] *= nfactor
        diagnorm[i + 2 * n0] *= nfactor
        diagnorm[i + 3 * n0] *= nfactor

    # Each diagnorm[i + _ * n0] should be a random normal variable
    # diagnorm[i + _ * n0] 应该是一个标准正态变量
    chistat = sum(elt ** 2 for elt in diagnorm) # 计算标准化对角线和的平方和
    pvalue = 1 - chi2.cdf(chistat, df=4 * (n0 - 1)) # 计算卡方分布的p值
    return pvalue


def parse_multivariate_file(filename):
    """
    函数用于解析一个包含多个多元样本的文件
    每行代表一个多元样本，假定所有样本来自同一分布
    函数返回样本的期望标准差（sigma）和包含所有样本的列表
    Parse a file containing several multivariate samples.

    Input:
    - the file name of a file containing k lines
      - each line corresponds to a multivariate sample
      - the samples are all assumed to be from the same distribution

    Output:
    - sigma: the expected standard deviation of the samples
    - data: a Python list of length k, containing all the samples

    解析包含多个多元样本的文件。

    输入：
    - filename：包含k行的文件路径
      - 每行对应一个多元样本
      - 所有样本假定来自同一分布

    输出：
    - sigma：样本的期望标准差
    - data：一个长度为k的Python列表，包含所有样本
    """
    with open(filename) as f: # 打开文件进行读取
        sigma = 0 # 初始化期望标准差为0
        data = [] # 用于存储所有样本的列表
        while True:
            # Parse each line
            # 逐行读取文件中的样本
            line = f.readline()
            if not line:
                break  # EOF
            # 使用正则表达式拆分每行数据，假设数据以逗号或逗号加换行符为分隔符
            sample = re.split(", |,\n", line)
            sample = [int(elt) for elt in sample[:-1]] # 将每个元素转换为整数，去掉最后一个元素（可能是换行符）
            data += [sample] # 将当前样本添加到数据列表中
            sigma += sum(elt ** 2 for elt in sample) # 计算当前样本的平方和，并累加到sigma中
        # sigma is the expected sigma based on the samples
        # 计算期望标准差，sigma为所有样本的平方和的平均值的平方根
        sigma = sqrt(sigma / (len(data) * len(data[0])))
    return (sigma, data) # 返回标准差和样本数据


def test_sig(n=128, nb_sig=1000, perturb=False, level=0):
    """
    函数用于测试 Falcon 签名生成的输出。
    可以通过扰动 FFT 树来检查签名是否仍然符合预期分布，尤其是在扰动较大的情况下，diagcov 测试会检测到签名的分布偏差
    Test signatures output by a Python implementation of Falcon.
    This test allow to perturb the FFT by setting the rightmost node
    of the FFT tree (of the private key) to 0. One can check that, at
    least for moderate levels (0 to 4), the test will end up detecting
    (via diagcov) that the signatures output do not follow the correct
    distribution.

    Input:
    - n: the degree of the ring
    - nb_sig: number of signatures
    - perturb: if set to 1, one node in the FFT tree is set to 0
    - level: determines which node (the rightmost one at a given level)
      is set to 0

    测试 Falcon Python 实现输出的签名。
    此测试通过将 FFT 树（私钥）中的最右节点设置为 0 来扰动 FFT。可以检查，对于适度的扰动级别（0 到 4），
    测试最终会通过 diagcov 检测到签名输出不遵循正确的分布。

    输入：
    - n：环的度数
    - nb_sig：签名数量
    - perturb：如果设置为 True，则将 FFT 树中的一个节点设置为 0
    - level：确定哪个节点（给定级别的最右节点）设置为 0
    """
    start = time.time() # 记录生成私钥开始时间
    # Generate a private key
    sk = falcon.SecretKey(n) # 生成私钥
    # Perturb the FFT tree
    # 扰动 FFT 树
    if perturb is True:
        # Check that the level is less than the FFT tree depth
        # 检查级别是否小于 FFT 树的深度
        assert(1 << level) < n
        u, k = sk.T_fft, n
        # Find the node
        # 查找需要设置为 0 的节点
        for _ in range(level):
            u = u[2] # 进入下一层
            k >>= 1 # 将级别减半
        # Zero-ize the node
        # 将节点置为 0
        u[0] = [0] * k
    end = time.time() # 记录生成私钥的结束时间
    print("Took {t:.2f} seconds to generate the private key.".format(t=end - start))

    # Compute signatures
    # 计算签名
    message = "0" # 用于签名的消息
    start = time.time() # 记录生成签名开始时间
    # 生成多个签名
    list_signatures = [sk.sign(message, reject=False) for _ in range(nb_sig)]
    # Strip away the nonces and concatenate the s_1's and s_2's
    # 去除随机数部分并连接 s_1 和 s_2 部分
    list_signatures = [sig[1][0] + sig[1][1] for sig in list_signatures]
    end = time.time() # 记录生成签名的结束时间
    print("Took {t:.2f} seconds to generate the samples.".format(t=end - start))
    # Perform the statistical test
    # 执行统计学测试
    start = time.time() # 记录运行统计测试的开始时间
    samples_data = MultivariateSamples(sk.sigma, list_signatures)
    end = time.time() # 记录运行统计测试的结束时间
    print("Took {t:.2f} seconds to run a statistical test.".format(t=end - start))
    return sk, samples_data # 返回私钥和样本数据


#######################
# Supplementary Stuff #
#######################

def qqplot(data):
    """
    https://www.itl.nist.gov/div898/handbook/eda/section3/qqplot.htm
    生成qq图，测试数据的多元正态性
    """
    data = pandas.DataFrame(data)
    data = deepcopy(data) # 创建数据的深拷贝

    # 计算样本协方差矩阵 S
    S = cov(data.transpose(), bias=1)
    n = len(data)
    p = len(data.columns)

    # 计算每个样本的均值
    means = [list(data.mean())] * n
    # 计算数据与均值的差异
    difT = data - pandas.DataFrame(means)
    # 计算差异矩阵的对角线元素 Dj
    Dj = diag(difT.dot(inv(S)).dot(difT.transpose()))
    # 计算样本数据与协方差矩阵 S 的逆的乘积 Y
    Y = data.dot(inv(S)).dot(data.transpose())
    # 获取 Y 的转置矩阵的对角线元素
    Ytdiag = array(pandas.DataFrame(diag(Y.transpose())))

    # 计算 Djk 矩阵
    Djk = - 2 * Y.transpose()
    Djk += tile(Ytdiag, (1, n)).transpose() # 按列复制 Ytdiag
    Djk += tile(Ytdiag, (1, n)) # 按行复制 Ytdiag

    # 快速提取 Djk 的值
    Djk_quick = []
    for i in range(n):
        Djk_quick += list(Djk.values[i])

    # 生成与样本数据维度一致的卡方分布随机数
    chi2_random = chi2.rvs(p - 1, size=len(Dj))
    chi2_random = sort(chi2_random) # 对卡方随机数进行排序
    # 计算样本的观测值与卡方分布随机数之间的线性回归 R^2 值
    r2 = stats.linregress(sort(Dj), sort(chi2_random))[2] ** 2

    # 设置图标题，显示 R^2 值
    plt.title('R-Squared = %0.20f' % r2, fontsize=9)
    plt.suptitle("QQ plot for Multivariate Normality", fontweight="bold", fontsize=12)

    # 保存图像并显示
    plt.savefig('qqplot.eps', format='eps', bbox_inches="tight", pad_inches=0)
    plt.show()
