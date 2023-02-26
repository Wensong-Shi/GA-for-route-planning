import csv
import random
import numpy as np
import math


def dataloader(filename):
    """
    读取一组GPS数据点
    :param filename: 存储GPS数据点的文件名
    :return: 存储GPS数据点的列表：[[lon1, lat1], [lon2, lat2], ...]
    """
    dataset = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            dataset.append(row)
    return dataset


def population_initial(scale, individual_length):
    """
    种群初始化
    :param scale: 种群规模
    :param individual_length: 个体基因编码长度
    :return: 初始化的种群，列表中的每个元素为一个个体
    """
    population = []
    primitive_individual = list(range(0, individual_length))
    for i in range(0, scale):
        random.shuffle(primitive_individual)
        population.append(primitive_individual)

    return population


def cal_distance(point1, point2):
    """
    用半正矢公式计算给定两点经纬度的距离
    :param point1:点1
    :param point2:点2
    :return:距离，单位：km
    """
    r = 6371.393  # 地球平均半径，单位：km
    # 弧度制
    lon1 = float(point1[0]) * math.pi / 180.0
    lat1 = float(point1[1]) * math.pi / 180.0
    lon2 = float(point2[0]) * math.pi / 180.0
    lat2 = float(point2[1]) * math.pi / 180.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # 半正矢公式
    h = (math.sin(dlat / 2) ** 2) + (math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    theta = math.asin(math.sqrt(h))
    d = 2 * r * theta

    return d


def distance_matrix(point_group):
    """
    计算距离矩阵
    :param point_group: 需计算的GPS数据点组
    :return: 距离矩阵：numpy二维数组，第i行j列的元素表示第i个点到第j个点的距离
    """
    n = len(point_group)
    d_matrix = np.zeros([n, n])
    for i in range(0, n):
        for j in range(0, n):
            d_matrix[i, j] = cal_distance(point_group[i], point_group[j])

    return d_matrix


def cal_fitness_i(individual, d_matrix):
    """
    计算单个个体的适应度
    :param individual: 单个个体
    :param d_matrix: 距离矩阵
    :return: 该个体的适应度
    """
    d_sum = 0
    for i in range(0, len(individual) - 1):
        flag1 = individual[i]
        flag2 = individual[i + 1]
        d_sum = d_sum + d_matrix[flag1, flag2]

    fitness_i = 1000.0 / d_sum
    return fitness_i


def cal_fitness(population, d_matrix):
    """
    计算种群的适应度
    :param population: 种群
    :param d_matrix: 距离矩阵
    :return: 种群的适应度列表，带标签：[[fitness0, 0], [fitness1, 1]...]
    """
    fitness = []
    for i in range(0, len(population)):
        fitness_i = cal_fitness_i(population[i], d_matrix)
        fitness.append([fitness_i, i])

    return fitness


def selection(population, fitness, ratio):
    """
    种群自然选择
    :param population: 种群
    :param fitness: 种群适应度
    :param ratio: 淘汰比例
    :return: 选择后的种群
    """
    fitness.sort(key=lambda x: x[0])  # 按适应度升序排序
    num = int(len(population) * ratio)  # 淘汰个体数
    # 获得适应度较低的个体对应的索引
    flag = []
    for fitness_i in fitness[:num]:
        flag.append(fitness_i[1])
    # 淘汰适应度较低的个体
    population_copy = population[:]
    for i in flag:
        population.remove(population_copy[i])

    return population


def crossover_prob(fitness):
    """
    根据适应度求交叉的概率
    :param fitness: 种群适应度
    :return: 种群中每个个体交叉的概率，带标签：[[prob0, 0], [prob1, 1]...]
    """
    sum_fit = 0
    for fitness_i in fitness:
        sum_fit = sum_fit + fitness_i[0]

    prob = []
    for fitness_i in fitness:
        prob_i = fitness_i[0] / sum_fit
        prob.append([prob_i, fitness_i[1]])
    return prob


def select_parents(population, prob, ratio, scale):
    """
    使用轮盘赌方式选择父代
    :param population: 种群
    :param prob: 交叉概率
    :param ratio: 淘汰比例
    :param scale: 种群规模
    :return: 父代，个数为scale * ratio * 2
    """
    num = scale * ratio * 2  # 父代个数
    parents = []
    parents_flag = []  # 存储种群中已经被选为父代的个体的索引
    while len(parents) < num:
        pointer = random.random()  # 指针
        sum_prob = 0  # 累计概率
        for i in range(0, len(population)):
            sum_prob = sum_prob + prob[i][0]
            if sum_prob >= pointer:
                if i in parents_flag:
                    break
                else:
                    parents.append(population[i])
                    parents_flag.append(i)
                    break

    return parents


def crossover(parents, length):
    """
    父代单点交叉生成子代
    :param parents: 父代
    :param length: 个体基因编码长度
    :return: 子代，数量为父代的1/2
    """
    break_point = math.ceil(length / 2)  # 交叉断点
    i = 0
    children = []
    while i < len(parents):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        child = parent1[:break_point]
        for j in parent2:
            if j not in child:
                child.append(j)

        children.append(child)
        i = i + 2

    return children


def mutation(children, pm, length):
    """
    子代发生变异
    :param children: 子代
    :param pm: 变异率
    :param length: 个体基因编码长度
    :return: 变异后的子代
    """
    for child in children:
        if random.random() <= pm:
            mutate_point1 = random.randint(0, length - 1)
            mutate_point2 = random.randint(0, length - 1)
            while mutate_point1 == mutate_point2:
                mutate_point2 = random.randint(0, length - 1)
            child[mutate_point1], child[mutate_point2] = child[mutate_point2], child[mutate_point1]

    return children


def find_best(fitness):
    """
    找到适应度最高的个体的索引
    :param fitness: 适应度
    :return: 适应度最高的个体的索引
    """
    fitness.sort(key=lambda x: x[0], reverse=True)  # 按适应度降序排序
    flag = fitness[0][1]

    return flag


def write_file(filename, data_list):
    """
    将数据列表逐个元素按行写入文件
    :param filename: 待写入的文件名
    :param data_list: 数据列表
    :return: 无
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for data in data_list:
            writer.writerow(data)
