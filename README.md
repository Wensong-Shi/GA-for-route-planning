# GA-for-route-planning
一个遗传算法实现，用于对一组GPS数据点进行线路规划，规划依据为线路路径最短。
## 文件说明
### point_group
.csv文件，存储需要线路规划的若干GPS数据点，每行两个字段，分别为一点的经纬度。  
该文件为附带的一组GPS数据点，可当作demo来运行处理。
## 算法说明
### 基本流程
初始化——计算适应度——选择算子——交叉算子——变异算子——回到第2步，直至迭代次数达到设定要求
### 超参数说明
#### scale
种群规模，一般应根据全体解的数量来设置。  
scale越大，计算可能越慢，但所需的迭代次数可能越少。
#### n
迭代次数。  
n越大，计算时间越长，但越有可能得到最优解。
#### pm
变异率。  
适当的pm可以保证算法跳出局部最优解，但pm过大可能阻碍最优解的产生。
#### elimination_ratio
淘汰率，每次迭代需淘汰掉scale*ratio个适应度较低的个体。  
在此实现中，ratio应小于1/3。
### 详细说明
#### 编码机制
假设一组待规划的GPS数据点数量为N，采用自然数编码机制，选择从1到N的自然数，令每种自然数的排列代表一种路径经过点的顺序，即一个可能解。  
如当N=5时，“51432”代表着“点5——点1——点4——点3——点2”这样一条线路。
#### 适应度
由于规划目标为线路路径最短，因此，若某个个体所代表的线路路径越长，其适应度越低。
#### 选择算子
在每次种群迭代的过程中，模拟自然选择的规律，按照淘汰比例淘汰掉若干个适应度较低的个体。
#### 交叉算子
对种群进行选择操作之后，在剩余个体中挑选一部分适应度较高的个体成为父代，从而进行交叉操作，产生若干个子代个体补充到种群中，以此来保证种群规模不变。  
选择父代的方式为轮盘赌选择，个体的适应度越高，被选为父代的概率越大。
#### 变异算子
在子代补充到种群之前，还要进行一次变异操作，目的是突变出一些新的基因型，防止算法收敛到局部解。  
具体的变异算子如下：根据变异率pm对子代中的每个个体进行一次判断，判断该个体是否发生变异，若是，则在1到N中随机选择两个不同的自然数i、j，之后交换该个体基因型中在第i个和第j个位置上的数字，完成变异。
