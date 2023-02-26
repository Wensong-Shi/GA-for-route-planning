# 使用遗传算法对GPS数据点进行路径规划
# 规划依据为线路路径最短
# 将规划好的线路按经过的GPS点顺序进行存储

import utils


# 读取一组GPS数据点，要求文件为.csv格式，每行两个字段，分别为一点的经纬度
filename = 'point_group'
point_group = utils.dataloader(filename)

# 超参数
scale = 100  # 种群规模
n = 100  # 迭代次数
pm = 0.8  # 变异率
elimination_ratio = 0.3  # 淘汰比例，应小于1/3

# 种群初始化
individual_length = len(point_group)  # 个体基因编码长度
population = utils.population_initial(scale, individual_length)  # 存储种群，列表中的每个元素为一个个体

d_matrix = utils.distance_matrix(point_group)  # 距离矩阵

# 种群迭代
for i in range(0, n):
    # 计算种群适应度
    fitness_all = utils.cal_fitness(population, d_matrix)
    # 根据适应度对种群进行选择
    population = utils.selection(population, fitness_all, elimination_ratio)

    # test
    print(f'The max fitness is {fitness_all[-1][0]}.')

    # 求剩余个体的适应度
    fitness_res = utils.cal_fitness(population, d_matrix)
    # 求剩余个体交叉的概率
    prob = utils.crossover_prob(fitness_res)
    # 根据轮盘赌方法选择父代
    parents = utils.select_parents(population, prob, elimination_ratio, scale)
    # 父代单点交叉生成子代
    children = utils.crossover(parents, individual_length)
    # 子代发生变异
    children = utils.mutation(children, pm, individual_length)
    # 形成新种群
    for child in children:
        population.append(child)

# 找到最优个体
fitness = utils.cal_fitness(population, d_matrix)
best_flag = utils.find_best(fitness)
best = population[best_flag]

# 对最优个体解码
route = []  # 按顺序存储GPS数据点，代表线路
for i in best:
    route.append(point_group[i])

# 存储线路
filename = 'route'
utils.write_file(filename, route)
