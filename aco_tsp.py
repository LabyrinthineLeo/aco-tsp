import math
import random, copy
from matplotlib import pyplot as plt


class SolveTSPUsingACO:
    class Edge:
        def __init__(self, a, b, weight, initial_pheromone):
            self.a = a
            self.b = b
            self.weight = weight  # 距离
            self.pheromone = initial_pheromone  # 信息素

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges):
            self.alpha = alpha  # 控制信息素浓度
            self.beta = beta  # 控制启发式信息
            self.num_nodes = num_nodes  # 节点数量
            self.edges = edges  # 节点信息矩阵
            self.tour = None  # 路径
            self.distance = 0.0  # 距离

        def _select_node(self):
            roulette_wheel = 0.0  # 轮盘赌的概率
            unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]
            heuristic_total = 0.0  # 启发式信息
            # 遍历未检索的节点，计算启发式信息
            for unvisited_node in unvisited_nodes:
                # 累加未到达节点距离信息
                heuristic_total += self.edges[self.tour[-1]][unvisited_node].weight
            # 计算累计所有未到达节点的信息和
            for unvisited_node in unvisited_nodes:
                roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
            # 随机生成阈值
            random_value = random.uniform(0.0, roulette_wheel)
            wheel_position = 0.0
            # 轮盘赌策略
            for unvisited_node in unvisited_nodes:
                wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                                  pow((heuristic_total / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)
                if wheel_position >= random_value:
                    return unvisited_node

        def find_tour(self):
            # 构建解路径
            self.tour = [random.randint(0, self.num_nodes - 1)]  # 随机初始化开始节点
            while len(self.tour) < self.num_nodes:
                # 按照选择概率构建路径
                self.tour.append(self._select_node())
            return self.tour

        def get_distance(self):
            # 计算路径长度
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
            return self.distance

    def __init__(self, mode='ACS', colony_size=10, elitist_weight=1.0, min_scaling_factor=0.001, alpha=1.0, beta=3.0,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, steps=100, nodes=None, labels=None):
        # ACO模式
        self.mode = mode
        # 蚁群数量
        self.colony_size = colony_size
        # 精英权重
        self.elitist_weight = elitist_weight
        # 最小尺度因子
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho
        # 信息素释放权重
        self.pheromone_deposit_weight = pheromone_deposit_weight
        # steps
        self.steps = steps
        # 节点数量
        self.num_nodes = len(nodes)
        # 节点信息
        self.nodes = nodes

        self.distance_list = []

        # 节点索引
        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.num_nodes + 1)

        # 城市关联矩阵
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        # 构建边信息类，包含物理距离和信息素
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j] = self.edges[j][i] = self.Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                                initial_pheromone)

        # 构建蚁群【self.edges是引用传递，局部和全局共享】
        # self.ants = [self.Ant(alpha, beta, self.num_nodes, copy.deepcopy(self.edges)) for _ in range(self.colony_size)]
        self.ants = [self.Ant(alpha, beta, self.num_nodes, self.edges) for _ in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def _add_pheromone(self, tour, distance, weight=1.0):
        # 信息素更新
        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def _acs(self):

        for step in range(self.steps):
            # 更新全局信息素
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

            # 遍历所有个体
            for ant in self.ants:
                # 解构建->信息素更新
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                # 更新全局解
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance

            # 更新全局信息素
            # for i in range(self.num_nodes):
            #     for j in range(i + 1, self.num_nodes):
            #         self.edges[i][j].pheromone *= (1.0 - self.rho)

            self.distance_list.append(self.global_best_distance)

    def _elitist(self):

        for step in range(self.steps):

            # 先挥发
            # for i in range(self.num_nodes):
            #     for j in range(i + 1, self.num_nodes):
            #         self.edges[i][j].pheromone *= (1.0 - self.rho)

            for ant in self.ants:
                self._add_pheromone(ant.find_tour(), ant.get_distance())
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance

            # 对最佳路径（精英蚂蚁）进一步增加信息素
            self._add_pheromone(self.global_best_tour, self.global_best_distance, weight=self.elitist_weight)

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)

            self.distance_list.append(self.global_best_distance)

    def _max_min(self):
        for step in range(self.steps):
            # 局部最优，每代的结果
            iteration_best_tour = None
            iteration_best_distance = float("inf")

            for ant in self.ants:
                # 解的构建
                ant.find_tour()
                # 更新局部最优
                if ant.get_distance() < iteration_best_distance:
                    iteration_best_tour = ant.tour
                    iteration_best_distance = ant.distance

            # 信息素挥发
            # for i in range(self.num_nodes):
            #     for j in range(i + 1, self.num_nodes):
            #         self.edges[i][j].pheromone *= (1.0 - self.rho)

            # 按照蚁群进化进度来确定更新全局最佳路径还是局部最佳路径
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour, iteration_best_distance)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_distance
            else:
                if iteration_best_distance < self.global_best_distance:
                    self.global_best_tour = iteration_best_tour
                    self.global_best_distance = iteration_best_distance
                self._add_pheromone(self.global_best_tour, self.global_best_distance)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_distance

            min_pheromone = max_pheromone * self.min_scaling_factor

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0 - self.rho)  # 其实是先累加再挥发
                    if self.edges[i][j].pheromone > max_pheromone:
                        self.edges[i][j].pheromone = max_pheromone
                    elif self.edges[i][j].pheromone < min_pheromone:
                        self.edges[i][j].pheromone = min_pheromone

            self.distance_list.append(self.global_best_distance)

    def run(self):
        print('Started : {0}'.format(self.mode))
        if self.mode == 'ACS':
            self._acs()
        elif self.mode == 'Elitist':
            self._elitist()
        else:
            self._max_min()
        print('Ended : {0}'.format(self.mode))
        print('Sequence : {0}'.format(' -> '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def plot_tour(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title(self.mode)
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        if save:
            if name is None:
                name = './tour_plots/ori/{0}_tour.png'.format(self.mode)
            plt.savefig(name, dpi=dpi)
        plt.show()
        plt.gcf().clear()

    def plot_opt(self):

        plt.figure(figsize=(12, 10))

        plt.plot(self.distance_list, label='Dist per step', color='b')
        plt.title('Train Curve')
        plt.xlabel('Step')
        plt.ylabel('Distance')

        # 显示图形
        # plt.show()
        plt.savefig('./tour_plots/ori/{0}_train.png'.format(self.mode))
        plt.gcf().clear()


if __name__ == '__main__':
    # 蚁群数量
    _colony_size = 5
    _steps = 200
    random.seed(10)
    _nodes = [(random.uniform(0, 200), random.uniform(0, 200)) for _ in range(0, 50)]
    print(_nodes)

    # 经典ACO
    acs = SolveTSPUsingACO(mode='ACS', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    acs.run()
    acs.plot_opt()
    acs.plot_tour()

    # 精英ACO
    elitist = SolveTSPUsingACO(mode='Elitist', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    elitist.run()
    elitist.plot_opt()
    elitist.plot_tour()

    # MaxMinACO
    max_min = SolveTSPUsingACO(mode='MaxMin', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    max_min.run()
    max_min.plot_opt()
    max_min.plot_tour()

