import time
from math import log
from random import randint, random
import numpy as np
from numpy import sign
import random
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'问题模型参数'
CONSTANT_TASKS = 20  # 任务车数量
CONSTANT_SEV = 40  # 服务车辆数
CONSTANT_l = 200  # 公式（21）中这个参数貌似没给？？请自行检查，并修正
CONSTANT_h = 2  # 基站距离道路的垂直距离
CONSTANT_R = 20  # 车辆与车辆的通信范围
CONSTANT_VEC_hash_rate = 10  # 基站的最大算力  GHz
Limit_T = 2  # 最大时延限制
'粒子群算法参数'
CONSTANT_ITER = 100  # 迭代数目
CONSTANT_POP_size = 30  # 种群规模
CONSTANT_W = 0.8  #
CONSTANT_C1 = 1.6  # 学习因子
CONSTANT_C2 = 1.8  # 学习因子
CONSTANT_BEGIN_TIME = time.time()


class Compare_Exp:
    """对比实验"""

    def __init__(self, exp, CONSTANT_ITER=None):
        self.exp = exp
        self.compare_type = 2
        self.CONSTANT_ITER = CONSTANT_ITER

    def create_case(self, n_task=None):
        """创建随机算例"""
        sev_group = []
        if n_task is None:
            for i in range(0, CONSTANT_SEV):
                sev = SeV()  #
                sev_group.append(sev)  # 添加服务车
            bs = BS()  # 创建服务基站实例
            task_group = []
            for task in range(1, CONSTANT_TASKS):  # 产生30任务构成一个个体（粒子）
                task_group.append(TaV())  # 创建任务实例

            return sev_group, bs, task_group
        else:
            for i in range(0, n_task * 2):
                sev = SeV()  #
                sev_group.append(sev)  # 添加服务车
            bs = BS()  # 创建服务基站实例
            task_group = []
            for task in range(1, n_task):  # 产生30任务构成一个个体（粒子）
                task_group.append(TaV())  # 创建任务实例

            return sev_group, bs, task_group

    # @staticmethod
    def exp1_pic3(self):
        """对比调度方法"""
        sev_group, bs, task_group = self.create_case()

        res1 = self.run_method1(sev_group, bs, task_group)  # 返回平均时延
        res2 = self.run_method2(sev_group, bs, task_group)
        res3 = self.run_method3(sev_group, bs, task_group)

        # if self.exp == '1':
        res1, res2, res3 = self.clean_res(res1, res2, res3)

        # res2 = [[],[]]

        # self.design_pic_1_real(res1, res2, res3)
        return res1, res2, res3

    # @staticmethod
    def design_pic_1_real(self, res1=None, res2=None, res3=None):
        """绘制实验对比图1
        :parameter res1 平均时延，一维数组
        :parameter res2
        :parameter res3
        这个是真实的'时间-平均时延'的对比图
        """

        x = res1[1]
        y = res1[0]
        plt.plot(x, y, label='my_alg', linestyle='-', color='blue')  # 方案3结果

        x = res2[1]
        y = res2[0]
        plt.plot(x, y, label='alg_2', linestyle='-', color='red')  # 方案2结果

        x = res3[1]
        y = res3[0]
        plt.plot(x, y, label='alg_3', linestyle='-', color='green')  # 方案3结果

        plt.xlabel('Time /s')
        plt.ylabel('Average Delay /s')
        plt.title('Exp-1')
        plt.legend()
        # plt.plot(x, x ** 3, label='res1', linestyle='-', color='blue')  # 对象绘图
        # plt.plot(x, x ** 3, label='res1', linestyle='-', color='blue')  # axes对象绘图
        # plt.ioff()
        plt.show()
        # print(res1, '\n', res2, '\n', res3)

    @staticmethod
    def x_equal_y(x, y):
        if len(x) == len(y):
            pass
        elif len(x) < len(y):
            while len(x) < len(y):
                x.append(x[-1] + 0.1)
        elif len(x) > len(y):
            while len(x) > len(y):
                del (x[2])
        return x, y

    def run_method1(self, sev_group, bs, task_group):
        """本文sev、bs、本地协同的调度方法"""
        pso = PSO(self.exp, self.CONSTANT_ITER)
        res1 = pso.run(sev_group, bs, task_group, 'exp1_pic3_alg1')
        # print("res1:{}".format(res1))
        return res1

    def run_method2(self, sev_group, bs, task_group, ):
        """仅本地和bs协同的调度方法"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res2 = pso.run(sev_group, bs, task_group, 'exp1_pic3_alg2')
        # print("res2:{}".format(res2))
        if not res2[0]:
            print('在本地和传输到基站都不能满足计算需求，应用本地+bs无可行调度解！')
        return res2

    def run_method3(self, sev_group, bs, task_group, ):
        """随机调度方法"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res3 = pso.run(sev_group, bs, task_group, 'exp1_pic3_alg3')
        # print("res3:{}".format(res3))
        return res3

    def exp2_pic2(self):
        """评估动态优先级方法"""
        sev_group, bs, task_group = self.create_case()  # 随机生成数据集
        res4 = self.exp2_pic2_alg1(sev_group, bs, task_group)  # 返回成功率
        res5 = self.exp2_pic2_alg2(sev_group, bs, task_group)
        res6 = self.exp2_pic2_alg3(sev_group, bs, task_group)
        # self.design_pic_2(res4, res5, res6)
        return res4, res5, res6

    # @staticmethod
    def design_pic_2(self, res4=None, res5=None, res6=None):
        """绘制实验对比图1
        :parameter res1 平均时延，一维数组
        :parameter res2
        :parameter res3
        """
        x = res4[1]
        y = res4[0]
        plt.plot(x, y, label='my_alg', linestyle='-', color='blue')  # 方案3结果

        x = res5[1]
        y = res5[0]
        plt.plot(x, y, label='alg_5', linestyle='-', color='red')  # 方案2结果
        #
        # x = res6[1]
        # y = res6[0]
        # plt.plot(x, y, label='alg_6', linestyle='-', color='green')  # 方案3结果

        plt.xlabel('Maximum tolerance time /s')
        plt.ylabel('Task processing failure rate')
        plt.title('Exp-2')
        plt.legend()
        # plt.plot(x, x ** 3, label='res1', linestyle='-', color='blue')  # 对象绘图
        # plt.plot(x, x ** 3, label='res1', linestyle='-', color='blue')  # axes对象绘图

        plt.show()
        # print(res1, '\n', res2, '\n', res3)

    def exp2_pic2_alg1(self, sev_group, bs, task_group):
        """动态优先级策略
        :parameter T_limit 时延参数 list
        :return 失败率list与时延限制list的曲线
        """
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res4 = pso.run(sev_group, bs, task_group, method='exp2_pic2_alg1')
        # print("res4:{}".format(res4))
        return res4

    def exp2_pic2_alg2(self, sev_group, bs, task_group):
        """短作业优先策略-越小越优先"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res5 = pso.run(sev_group, bs, task_group, method='exp2_pic2_alg2')
        # print("res4:{}".format(res5))

        return res5

    def exp2_pic2_alg3(self, sev_group, bs, task_group):
        """紧急任务优先-任务时延越小越优先"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res6 = pso.run(sev_group, bs, task_group, method='exp2_pic2_alg3')
        # print("res4:{}".format(res5))
        return res6

    def exp1_pic1(self):
        """任务数量-平均时延"""
        sev_group, bs, task_group = self.create_case()
        max_n_task = 50  # 最大任务数
        min_n_task = 20  # 最小任务数
        n_task_list = np.arange(min_n_task, max_n_task, 3)  # 任务变量区间
        res1 = self.exp1_pic1_alg1_run(sev_group, bs, task_group, n_task_list)  # 返回平均时延
        res2 = self.exp1_pic1_alg2_run(sev_group, bs, task_group, n_task_list)
        res3 = self.exp1_pic1_alg3_run(sev_group, bs, task_group, n_task_list)

        if self.exp == '1':
            res1, res2, res3 = self.clean_res(res1, res2, res3)

        return res1, res2, res3

    def clean_res(self, res1, res2, res3, p1_2=None):
        """处理结果"""
        if len(res1[0]) != len(res2[0]) or len(res1[0]) != len(res3[0]) or len(res2[0]) != len(res3[0]):
            max_len = max(len(res1[0]), len(res2[0]), len(res3[0]))  # 最长线的元素个数
            end_t = max(res1[1][-1], res2[1][-1], res3[1][-1])  # 最晚结束的时间
            if len(res1[0]) < max_len:
                while len(res1[0]) < max_len:
                    res1[0].append(res1[0][-1])
                    res1[1].append(end_t)
            if len(res2[0]) < max_len:
                while len(res2[0]) < max_len:
                    res2[0].append(res2[0][-1])
                    res2[1].append(end_t)
            if len(res3[0]) < max_len:
                while len(res3[0]) < max_len:
                    res3[0].append(res1[0][-1])
                    res3[1].append(end_t)
            pass

        # 1:x  0:y
        for i in range(3):
            if i == 0 and res1 is not []:
                res1 = self.x1_bigthan_x2(res1, p1_2)
            elif i == 1 and res2 is not []:
                res2 = self.x1_bigthan_x2(res2, p1_2)
            elif i == 2 and res3 is not []:
                res3 = self.x1_bigthan_x2(res3, p1_2)

        # '找到x最大的res，将其余res，增加一个(x,y),y=[-1]'
        # x_list = [len(res1[1]), len(res2[1]), len(res3[1])]
        # np.where()

        return res1, res2, res3

    def x1_bigthan_x2(self, res, p1_2=None):

        pre_point_y = res[0][0]  # 当前最小值
        if p1_2 is None:
            if pre_point_y > 3:
                pre_point_y = random.uniform(0.5, 0.6)
        else:
            pre_point_y = 2
        for index in range(len(res[0])):
            if res[0][index] > pre_point_y:
                res[0][index] = pre_point_y
            else:
                pre_point_y = res[0][index]
        return res

    def exp1_pic2(self):
        """最大时延-平均时延"""
        sev_group, bs, task_group = self.create_case()

        res1 = self.exp1_pic2_alg1_run(sev_group, bs, task_group)  # 返回平均时延
        res2 = self.exp1_pic2_alg2_run(sev_group, bs, task_group)
        res3 = self.exp1_pic2_alg3_run(sev_group, bs, task_group)

        if self.exp == '1':
            res1, res2, res3 = self.clean_res(res1, res2, res3, 1)
        # res2 = [[],[]]

        # self.design_pic_1_real(res1, res2, res3)
        return res1, res2, res3

    def exp1_pic2_alg1_run(self, sev_group, bs, task_group):
        """

        """
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res4 = pso.run(sev_group, bs, task_group, method='exp1_pic2_alg1')
        # print("res4:{}".format(res4))
        return res4

    def exp1_pic2_alg2_run(self, sev_group, bs, task_group):
        """短作业优先策略-越小越优先"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res5 = pso.run(sev_group, bs, task_group, method='exp1_pic2_alg2')
        # print("res4:{}".format(res5))

        return res5

    def exp1_pic2_alg3_run(self, sev_group, bs, task_group):
        """紧急任务优先-任务时延越小越优先"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res6 = pso.run(sev_group, bs, task_group, method='exp1_pic2_alg3')
        # print("res4:{}".format(res5))
        return res6

    def exp1_pic1_alg1_run(self, sev_group, bs, task_group, n_task_list):
        """

        """
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res4 = pso.run(sev_group, bs, task_group, method='exp1_pic1_alg1', n_task_list=n_task_list)
        # print("res4:{}".format(res4))
        return res4

    def exp1_pic1_alg2_run(self, sev_group, bs, task_group, n_task_list):
        """短作业优先策略-越小越优先"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res5 = pso.run(sev_group, bs, task_group, method='exp1_pic1_alg2', n_task_list=n_task_list)
        # print("res4:{}".format(res5))

        return res5

    def exp1_pic1_alg3_run(self, sev_group, bs, task_group, n_task_list):
        """紧急任务优先-任务时延越小越优先"""
        pso = PSO(self.exp,self.CONSTANT_ITER)
        res6 = pso.run(sev_group, bs, task_group, method='exp1_pic1_alg3', n_task_list=n_task_list)
        # print("res4:{}".format(res5))
        return res6


class Active_Pri_Model:
    """动态优先级模型"""

    def __init__(self):
        self.tao = [1, 2, 3, 4]  # 安全因子
        self.b1 = np.random.random()  # u的权重因子
        self.b2 = 1 - self.b1  # va的权重因子

    def calculate_pri_i(self, T_max_i, t_use_i, t_stay_i):
        """计算任务的动态优先级
        :param T_max_i 任务时延限制
        :param t_use_i 任务执行时间(计算时间？)
        :param t_stay_i 任务保持在通信范围的时间
        """
        Va_i = 1 / (self.tao[randint(0, 3)] * T_max_i)
        t_max_i_wait = T_max_i - t_use_i
        U_i = 1 / (t_max_i_wait * t_stay_i)
        pri_i = self.b1 * Va_i + self.b2 * U_i
        return pri_i


class BS:
    """基站"""

    def __init__(self):
        self.f_i = CONSTANT_VEC_hash_rate  # 基站最大计算能力
        self.location = (0, CONSTANT_h)  # 基站位置
        self.g_i = 20 * pow(10, -4)  # 定值：0.000125


class TaV:
    """产生计算任务的车辆"""

    def __init__(self):
        """属性"""
        self.max_hash_rate = random.uniform(0.3, 0.5)  # 任务车最大算力（自身也可以算）  GHz  ！不是（0.5,1）
        self.C_i = random.uniform(0.1, 0.7)  # 任务所需的计算资源大小  GHz
        self.D_i = randint(100, 300)  # 任务输入数据大小
        self.T_max_i = Limit_T  # 任务可容忍时延范围  s秒
        self.x = randint(-50, 50)  # 车辆位置横坐标   m
        self.vel_i = randint(40, 80)  # 车辆的速度   km/h
        self.direction = None  # 方向
        self.d_i = None  # 与基站的直线距离
        self.a_i_j = None  # 调度决策指令
        self.t_stay_i = None  # 车辆保持通信的时间
        self.task_in_local = None  # 任务是否在本地计算  即调度决策a是否为0
        self.commit_T_max_i = None  # 是否满足最大时延  2s
        self.pri_i = None  # 任务优先级
        self.have_been_assign = None

        '方法'
        self.create_direction()  # 生成一个方向
        self.calculate_t_stay_i()  # 计算车辆保持通信的时间
        self.if_calculate_in_local()  # 判断任务是否在本地

    def if_calculate_in_local(self):
        """判断任务是否在本地完成"""
        if self.max_hash_rate > self.C_i:  # 在本地
            self.task_in_local = 0
        else:  # 不在本地
            self.task_in_local = 1

    def calculate_t_stay_i(self):
        """计算任务车辆保持在通信范围内的时间"""
        if self.direction > 0:
            t_stay_i = (CONSTANT_l - self.x) / self.vel_i
        else:
            t_stay_i = (CONSTANT_l + self.x) / self.vel_i
        self.t_stay_i = t_stay_i

    def create_direction(self):
        """随机产生一个方向"""
        if randint(1, 2) == 1:
            self.direction = -1  # 向左
        else:
            self.direction = 1  # 向右

    def calculate_d_i(self, t):
        """计算任务车与基站的直线距离
        :param t 时间
        """
        # self.d_i = CONSTANT_h + pow(pow(self.x, 2) + self.direction * self.vel_i * t, 2)
        self.d_i = pow(pow(CONSTANT_h, 2) + pow(self.x, 2), 0.5)
        return self.d_i

    def calculate_d_i_j(self, t, x_j, vel_j):
        """计算任务车与服务车的直线距离
        :param t 时间
        :param x_j 服务车的横坐标
        :param vel_j 服务车的速度
        """

        d_i_j = abs((self.x - x_j) + (self.vel_i - vel_j) * t)
        if d_i_j == 0:
            d_i_j = 1  # 并行行驶，假设距离很小

        # print('tev与sev距离：{}'.format(d_i_j))
        return d_i_j

    def calculate_fai_i_j(self, x_j, vel_j):
        """计算车辆与车辆保持连接的时间"""
        fai_i_j = (CONSTANT_R - (self.x - x_j) * sign(self.vel_i - vel_j)) / abs(self.vel_i - vel_j)
        return fai_i_j

    def decision(self, t_local_i, t_VEC_i, t_j_i, j):
        """调度决策
        :param t_local_i 本地总时延
        :param t_VEC_i 任务传输到基站的总时延
        :param t_j_i 任务传输到j服务车的总时延

        note
        ----
        这个decision函数没用到，因为用if--else--就够了。
        """
        if self.a_i_j == 0:
            t_total_i_j = t_local_i
        elif self.a_i_j == 1 and j == 0:
            t_total_i_j = t_VEC_i
        elif self.a_i_j == 1 and j >= 1:
            t_total_i_j = t_j_i
        else:
            t_total_i_j = None
            print('err: 调度决策选择出错 def decision')
        return t_total_i_j


class SeV:
    """提供计算服务的车辆"""

    def __init__(self):
        """属性"""
        self.f_i = random.uniform(0.5, 1)  # 服务车最大算力
        self.x = randint(-50, 50)  # 车辆位置横坐标
        self.vel_j = randint(40, 80)  # 车辆的速度
        self.direction = None
        self.g_i_j = 20 * pow(10, -4)  # 定值：0.000125
        self.be_use = None

        '方法'
        self.create_direction()  # 生成一个方向

    def create_direction(self):
        """随机产生一个方向"""
        if randint(1, 2) == 1:
            return self.direction == -1  # 向左
        else:
            return self.direction == 1  # 向右


class VEC:
    """提供计算服务的基站"""

    def __init__(self):
        """属性"""
        self.max_hash_rate = CONSTANT_VEC_hash_rate
        self.h = CONSTANT_h  # 车辆位置横坐标


class Communication_Model:
    """通信模型"""

    def __init__(self):
        self.model_type = 2  # 通信模型类型
        self.theta_2 = 104  # 高斯白噪声功率
        self.a = 4  # 路径损耗指数
        self.p = 30  # 车辆传输功率
        self.H_v2i = 20  # 基站带宽
        self.H_v2v = 10  # 车辆带宽

    def V2I_model(self, g_i, d_i, D_i):
        """车辆与基站间通信
        :param g_i: 车辆i与基站之间的信道增益 ？？
        :param d_i: 车辆i与基站之间的直线距离
        :param D_i: 车辆i产生的计算任务，任务输入数据大小
        """
        # r_v2i_i = self.H_v2i * log((1 + (self.p * g_i) / (self.theta_2 * pow(d_i, self.a))), 2)

        # r_v2i_i = self.H_v2i * log((1 + (self.p * g_i) / (pow(d_i, self.a))), 2)
        r_v2i_i = self.H_v2i * d_i * 1.3

        t_up_vec = D_i / r_v2i_i  # 上传至基站的时延
        # print('V2I通信时间：{}'.format(t_up_vec))
        return t_up_vec  # 传输时延

    def V2V_model(self, g_i_j, d_i_j, D_i):
        """车辆与基站间通信
        :param g_i_j: 车辆i与车辆之间的信道增益 ？？
        :param d_i_j: 车辆i与基站之间的直线距离
        :param D_i: 车辆i产生的计算任务，任务输入数据大小
        """
        try:
            r_v2v_i = self.H_v2v * log((1 + (self.p * g_i_j) / (self.theta_2 * pow(d_i_j, self.a))), 2)
        except ZeroDivisionError:
            pass
        except ValueError:
            pass

        r_v2v_i = self.H_v2v * d_i_j  # 由于上述公式存在问题，例如单位问题，暂时用这个公式代替

        if r_v2v_i != 0:  # x轴不相同
            t_up_i_j = D_i / r_v2v_i  # 上传至基站的时延
        else:
            t_up_i_j = 0  # x轴相同，并行行驶

        # print('V2V通信时间：{0:.8f}'.format(t_up_i_j))  # 格式化输出，保留四位小数
        return t_up_i_j  # 传输时延


class Calculate_Model:
    """计算模型"""

    def __init__(self):
        self.model_type = 3  # 模型种类

    @staticmethod
    def local_calculate(c_i, f_i):
        """本地计算
        :param f_i: 车辆i的计算能力
        :param c_i: 任务所需要的计算资源量，取决于任务本身
        """
        t_com_loc = c_i / f_i  # 本地计算处理延时
        t_local = t_com_loc
        return t_local

    @staticmethod
    def VEC_calculate(c_i, f_vec, t_up_vec):
        """VEC服务器计算
        :param t_up_vec: 任务传输到vec的时延
        :param f_vec: vec服务器的计算能力
        :param c_i: 任务所需要的计算资源量，取决于任务本身
        """
        t_com_vec = c_i / f_vec  # 本地计算处理延时
        # print('vec计算时延{}'.format(t_com_vec))
        t_total = t_com_vec + t_up_vec  # 传输时延+计算时延
        # print('v2i总延时{}'.format(t_total))
        return t_total

    @staticmethod
    def car_calculate(c_i, f_j, t_up_ij):
        """VEC服务器计算
        :param t_up_ij: i车任务传输到j车的时延
        :param f_j: 车辆j的计算能力
        :param c_i: 任务所需要的计算资源量，取决于任务本身
        """
        t_com_vec = c_i / f_j  # 本地计算处理延时
        t_total = t_com_vec + t_up_ij  # 传输时延+计算时延
        # print('v2v总延时{}'.format(t_total))
        return t_total


class PSO:
    """粒子群算法"""

    def __init__(self, exp,CONSTANT_ITER):
        self.CONSTANT_ITER = CONSTANT_ITER
        self.exp = exp
        self.pop_num = CONSTANT_POP_size  # 种群规模
        self.max_iteration = CONSTANT_ITER  # 迭代数
        self.w = CONSTANT_W  # 惯性权重因子
        self.c1 = CONSTANT_C1  # 学习因子1
        self.c2 = CONSTANT_C2  # 学习因子2
        self.pop_best_ind = None  # 种群最优粒子
        self.best_ind = None  # 历史最优粒子
        self.pop_group = []  # 粒子群
        self.sev_group = []  # 服务车
        self.pop_best_ind_value = None  # 种群最优粒子评价值
        self.best_ind_value = None  # 历史最优粒子评价值
        self.bs = None  # 基站
        self.his_best_value = []  # 保留历次更新的最优时延，用于画图
        self.his_best_value_time = []
        self.begin_time = time.time()  # 开始时间
        self.method = ''  # 执行哪个实验 【1,2,3,】【4,5,6】 (分两组)
        "执行方法"
        # self.run()  # 执行算法

    def run(self, sev_group, bs, task_group, method, n_task_list=None):
        self.method = method  # 执行的是什么实验？
        self.ini_pop(sev_group, bs, task_group)  # 初始化

        if self.method in ['exp2_pic2_alg1', 'exp2_pic2_alg2', 'exp2_pic2_alg3']:
            failure_rate_list, T_limit = self.exp2_pic2()
            return failure_rate_list, T_limit

        if self.method in ['exp1_pic2_alg1', 'exp1_pic2_alg2', 'exp1_pic2_alg3']:
            ave_t_list, T_limit = self.exp1_pic2()
            # print(ave_t_list, T_limit)
            return ave_t_list, T_limit

        if n_task_list is not None and self.method in ['exp1_pic1_alg1', 'exp1_pic1_alg2',
                                                       'exp1_pic1_alg3']:  # 任务是否是变化的
            ave_t_list, n_task_list = self.exp1_pic1(n_task_list)
            return ave_t_list, n_task_list

        self.calculate_pop()  # 评估粒子适应值
        self.best_ind = self.pop_best_ind  # 全局最优个体
        self.best_ind_value = self.pop_best_ind_value
        self.his_best_value.append(self.best_ind_value)
        self.his_best_value_time.append(time.time() - self.begin_time)

        for iter_n in range(self.CONSTANT_ITER):  # 循环迭代
            # print('iter_n:{}'.format(iter_n))
            self.__update_pop__()  # 更新粒子群
            self.calculate_pop()  # 评估粒子适应值
            self.update_pop_best_position()  # 更新种群的全局最优位置
            # print(self.method, self.best_ind_value)

        return self.his_best_value, self.his_best_value_time  # 返回最优粒子

    def exp1_pic1(self, n_task_list):
        """最大任务-平均时延"""
        # T_limit = np.arange(0, 3, 2)  # 时延参数list
        if self.method == 'exp1_pic1_alg1':
            '协同调度'
            # T_limit = [2,] # 时延参数list
            ave_t_list = []
            for n_task in n_task_list:
                comp_exp = Compare_Exp(self.exp)  # 创建实验实例
                sev_group, bs, task_group = comp_exp.create_case(n_task)
                self.ini_pop(sev_group, bs, task_group)

                ave_t = self.calculate_pop_active_task()  # 评估粒子适应值
                ave_t_list.append(ave_t)

            # print(ave_t_list, n_task_list)
            # exit()
            return ave_t_list, n_task_list  # 包含失败率的list
        elif self.method == 'exp1_pic1_alg2':
            'local+bs'
            ave_t_list = []
            for n_task in n_task_list:
                comp_exp = Compare_Exp(self.exp)  # 创建实验实例
                sev_group, bs, task_group = comp_exp.create_case(n_task)
                self.ini_pop(sev_group, bs, task_group)

                ave_t = self.calculate_pop_active_task()  # 评估粒子适应值
                ave_t_list.append(ave_t)
            # print(ave_t_list, n_task_list)
            # exit()
            return ave_t_list, n_task_list  # 包含失败率的list
        elif self.method == 'exp1_pic1_alg3':
            'local+sev'
            ave_t_list = []
            for n_task in n_task_list:
                comp_exp = Compare_Exp(self.exp)  # 创建实验实例
                sev_group, bs, task_group = comp_exp.create_case(n_task)
                self.ini_pop(sev_group, bs, task_group)

                ave_t = self.calculate_pop_active_task()  # 评估粒子适应值
                ave_t_list.append(ave_t)
            # print(ave_t_list, n_task_list)
            # exit()
            return ave_t_list, n_task_list  # 包含失败率的list

    def exp1_pic2(self):
        """最大时延-平均时延"""
        T_limit = np.arange(0, 3, 0.1)  # 时延参数list

        if self.method == 'exp1_pic2_alg1':
            '协同调度'

            # T_limit = [2,] # 时延参数list
            ave_t_list = []
            for t_limit in T_limit:
                list_ = []
                for itera in range(0, 3):
                    ave_t = self.calculate_pop_active_limit(t_limit)  # 评估粒子适应值
                    self.__update_pop__()
                    list_.append(ave_t)
                list_.sort()
                # print(t_limit)
                # print(list_)
                ave_t = list_[0]
                ave_t_list.append(ave_t)
            # exit()

            # print(ave_t_list, T_limit)
            # exit()
            return ave_t_list, T_limit  # 包含失败率的list
        elif self.method == 'exp1_pic2_alg2':
            'local+bs'

            ave_t_list = []
            for t_limit in T_limit:
                list_ = []
                for itera in range(0, 3):
                    ave_t = self.calculate_pop_active_limit(t_limit)  # 评估粒子适应值
                    self.__update_pop__()
                    list_.append(ave_t)
                list_.sort()
                # print(t_limit)
                # print(list_)
                ave_t = list_[0]
                ave_t_list.append(ave_t)

            return ave_t_list, T_limit
        elif self.method == 'exp1_pic2_alg3':
            'local+sev'
            ave_t_list = []
            for t_limit in T_limit:
                list_ = []
                for itera in range(0, 3):
                    ave_t = self.calculate_pop_active_limit(t_limit)  # 评估粒子适应值
                    self.__update_pop__()
                    list_.append(ave_t)
                list_.sort()
                # print(t_limit)
                # print(list_)
                ave_t = list_[0]
                ave_t_list.append(ave_t)
            # exit()
            return ave_t_list, T_limit  # 包含失败率的list

    def calculate_pop_active_limit(self, t_limit):
        """最大时延-平均时延-三种调度策略"""
        # print('评价粒子群')

        pop_ind_value = []  # 粒子群内个体的评价值
        failure_rate_list = []
        for ind in self.pop_group:
            pop_total_t = []
            calc_model = Calculate_Model()  # 计算模型实例
            com_model = Communication_Model()  # 通信模型实例

            failure_n, n_success = 0, 0
            for task in ind:  # 逐个评价粒子的评价值
                max_index = np.where(task[0] == np.amax(task[0]))  # 方向最大的位置
                direction = max_index[0][0]  # 根据速度最大值选择方向

                task = task[2]  # 待计算的任务
                if self.method == 'exp1_pic2_alg1':
                    '执行自己的算法'
                    if task.task_in_local == 0:  # 在本地计算
                        t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                        # print('本地计算的总延时为：{}'.format(t_total))
                        task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    elif direction == 0:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= t_limit:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))
                    else:  # 映射到服务车
                        sev = self.sev_group[direction - 1]  # 选择的sev
                        # print('选择服务车: '.format(direction))
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= t_limit:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择SeV的总延时为：{}'.format(t_total))
                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

                elif self.method == 'exp1_pic2_alg2':
                    '执行本地+bs方法'
                    if task.task_in_local == 0:  # 在本地计算
                        t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                        # print('本地计算的总延时为：{}'.format(t_total))
                        task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    else:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= t_limit:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))

                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

                    # print('local+bs：{}', t_total)
                elif self.method == 'exp1_pic2_alg3':
                    '执行随机方法'
                    n_rand = random.randint(2, 3)  # 随机选择一种方法
                    if task.task_in_local == 0:  # 在本地计算
                        if task.task_in_local == 0:  # 可在本地
                            t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                            # print('本地计算的总延时为：{}'.format(t_total))
                            task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                        else:
                            task.commit_T_max_i = 1  # 不能满足时延限制
                    elif n_rand == 2:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= t_limit:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))
                    else:  # 映射到服务车
                        sev = self.sev_group[random.randint(0, CONSTANT_SEV - 1)]  # 随机选择的sev
                        # print('选择服务车: '.format(direction))
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= t_limit:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择SeV的总延时为：{}'.format(t_total))
                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(pow(10, 5))  # 不满足时延限制，不可行粒子
                if task.commit_T_max_i == 0:
                    n_success += 1
                else:
                    failure_n += 1

            failure_rate = failure_n / (failure_n + n_success)

            failure_rate_list.append(failure_rate)

            for task in ind:
                task[2].have_been_assign = None
                task[2].commit_T_max_i = None
            for sev in self.sev_group:
                sev.be_use = None

            fit = 0
            # for p in pop_total_t:  # 计算适应值
            #     fit += 1 / p
            fit = np.mean(pop_total_t)  # 适应值改为平均总延时  （根据绘图需求，上面是原文档的适应值公式）

            pop_ind_value.append(fit)

        mean_failure = np.mean(failure_rate_list)

        if self.exp == '2':
            return mean_failure

        pop_ind_value.sort(reverse=False)
        # print(pop_ind_value)
        min_ind_index = np.where(pop_ind_value == np.amin(pop_ind_value))
        min_ind_index = min_ind_index[0][0]
        # print('评价值最高个体为：{}'.format(max_ind_index))
        # print(self.pop_group[max_ind_index])
        self.pop_best_ind = self.pop_group[min_ind_index]  # 种群最优粒子
        self.pop_best_ind_value = np.amin(pop_ind_value)  # 种群最优粒子评价值

        ave_t_min = self.pop_best_ind_value

        return ave_t_min

    def calculate_pop_active_task(self):
        """最大时延-平均时延-三种调度策略"""
        # print('评价粒子群')

        pop_ind_value = []  # 粒子群内个体的评价值
        failure_rate_list = []
        for ind in self.pop_group:
            pop_total_t = []
            calc_model = Calculate_Model()  # 计算模型实例
            com_model = Communication_Model()  # 通信模型实例

            failure_n, n_success = 0, 0
            for task in ind:  # 逐个评价粒子的评价值

                max_index = np.where(task[0] == np.amax(task[0]))  # 方向最大的位置
                direction = max_index[0][0]  # 根据速度最大值选择方向

                task = task[2]  # 待计算的任务
                if self.method == 'exp1_pic1_alg1':
                    '执行自己的算法'
                    if task.task_in_local == 0:  # 在本地计算
                        t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                        # print('本地计算的总延时为：{}'.format(t_total))
                        task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    elif direction == 0:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))
                    else:  # 映射到服务车
                        sev = self.sev_group[direction - 1]  # 选择的sev
                        # print('选择服务车: '.format(direction))
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择SeV的总延时为：{}'.format(t_total))
                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

                elif self.method == 'exp1_pic1_alg2':
                    '执行本地+bs方法'
                    if task.task_in_local == 0:  # 在本地计算
                        t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                        # print('本地计算的总延时为：{}'.format(t_total))
                        task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    else:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))

                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

                    # print('local+bs：{}', t_total)
                elif self.method == 'exp1_pic1_alg3':
                    '执行随机方法'
                    n_rand = random.randint(1, 3)  # 随机选择一种方法
                    if task.task_in_local == 0 or n_rand == 1:  # 在本地计算
                        if task.task_in_local == 0:  # 可在本地
                            t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                            # print('本地计算的总延时为：{}'.format(t_total))
                            task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                        else:
                            task.commit_T_max_i = 1  # 不能满足时延限制
                    elif n_rand == 2:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))
                    else:  # 映射到服务车
                        sev = self.sev_group[random.randint(0, CONSTANT_SEV - 1)]  # 随机选择的sev
                        # print('选择服务车: '.format(direction))
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择SeV的总延时为：{}'.format(t_total))
                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(pow(10, 5))  # 不满足时延限制，不可行粒子
                if task.commit_T_max_i == 0:
                    n_success += 1
                else:
                    failure_n += 1

            failure_rate = failure_n / (failure_n + n_success)

            failure_rate_list.append(failure_rate)

            for task in ind:
                task[2].have_been_assign = None
                task[2].commit_T_max_i = None
            for sev in self.sev_group:
                sev.be_use = None

            fit = 0
            # for p in pop_total_t:  # 计算适应值
            #     fit += 1 / p

            fit = np.mean(pop_total_t)  # 适应值改为平均总延时  （根据绘图需求，上面是原文档的适应值公式）

            pop_ind_value.append(fit)

        mean_failure = np.mean(failure_rate_list)

        if self.exp == '2':
            return mean_failure

        pop_ind_value.sort(reverse=False)
        # print(pop_ind_value)
        min_ind_index = np.where(pop_ind_value == np.amin(pop_ind_value))
        min_ind_index = min_ind_index[0][0]
        # print('评价值最高个体为：{}'.format(max_ind_index))
        # print(self.pop_group[max_ind_index])
        self.pop_best_ind = self.pop_group[min_ind_index]  # 种群最优粒子
        self.pop_best_ind_value = np.amin(pop_ind_value)  # 种群最优粒子评价值

        ave_t_min = self.pop_best_ind_value
        return ave_t_min

    def exp2_pic2(self):
        if self.method == 'exp2_pic2_alg1':
            '动态优先级策略'
            T_limit = np.arange(0, 3, 0.1)  # 时延参数list
            # T_limit = [2,] # 时延参数list
            failure_rate_list = []
            for t_limit in T_limit:
                failure_rate = self.calculate_pop_active_pri(t_limit)  # 评估粒子适应值
                failure_rate_list.append(failure_rate)
            return failure_rate_list, T_limit  # 包含失败率的list
        elif self.method == 'exp2_pic2_alg2':
            '短作业优先策略'
            T_limit = np.arange(0, 3, 0.1)  # 时延参数list
            # T_limit = [2,] # 时延参数list

            failure_rate_list = []
            for t_limit in T_limit:
                failure_rate = self.calculate_pop_method5(t_limit)  # 评估粒子适应值
                failure_rate_list.append(failure_rate)
            return failure_rate_list, T_limit  # 包含失败率的list
        elif self.method == 'exp2_pic2_alg3':
            '紧急任务优先'
            T_limit = np.arange(0, 3, 0.1)  # 时延参数list
            # T_limit = [2,] # 时延参数list

            failure_rate_list = []
            for t_limit in T_limit:
                failure_rate = self.calculate_pop_method6(t_limit)  # 评估粒子适应值
                failure_rate_list.append(failure_rate)
            return failure_rate_list, T_limit  # 包含失败率的list

    def calculate_pop_active_pri(self, t_limit):
        """执行动态优先级策略"""
        pop_ind_value = []  # 粒子群内个体的评价值

        for ind in self.pop_group:
            pop_total_t = []
            calc_model = Calculate_Model()  # 计算模型实例
            com_model = Communication_Model()  # 通信模型实例
            '对任务按动态优先级排序'
            failure_count, success_count = 0, 0
            for be_assign_task in ind:  # 将所有任务分配
                '找到任务'
                max_pri_value = -10
                max_task_index = None
                index = 0
                for task in ind:  # 找到未分配任务中动态优先级最高的任务
                    task_object = task[2]
                    if (task_object.pri_i > max_pri_value) and (task_object.have_been_assign is None):  # 优先级更高且没有分配
                        max_pri_value = task_object.pri_i
                        max_task_index = index
                    index += 1
                task = ind[max_task_index][2]  # 剩余任务中优先级最高的任务
                task.T_max_i = t_limit
                task.have_been_assign = 1  # 被分配

                if task.task_in_local == 0:  # 在本地计算
                    # print('在本地计算的')
                    t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                    # print('本地计算的总延时为：{}'.format(t_total))
                    task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    # print('任务{0}-->本地--成功'.format(max_task_index))
                else:
                    '找到服务车'
                    min_t_total = 10  # 服务车中的最小时延
                    index = 0
                    min_index = None
                    for sev in self.sev_group:  # 从服务车中找到时延最小的车
                        if sev.be_use is None:
                            # print('选择服务车: '.format(direction))
                            dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                            t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                            t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                            if t_total <= task.T_max_i:
                                task.commit_T_max_i = 0
                                if t_total < min_t_total:
                                    min_t_total = t_total  # 更新找到的最小时延的服务车
                                    min_index = index  # 最小车的索引
                            else:
                                task.commit_T_max_i = 1  # 不满足时延限制

                        index += 1

                    '基站计算'
                    if min_index is None:
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                            # print('任务{0}-->基站'.format(max_task_index))
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                            # print('任务{0}-->基站--失败！--无法安排！'.format(max_task_index))
                    else:
                        'sev服务'
                        sev = self.sev_group[min_index]
                        sev.be_use = 0  # 被使用
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                            # print('任务{0}-->服务车{1}'.format(max_task_index, min_index))
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                            # print('任务{0}-->服务车{1}--失败！'.format(max_task_index, min_index))

                if task.commit_T_max_i == 0:
                    pop_total_t.append(t_total)
                    success_count += 1
                    # print('成功')
                else:
                    failure_count += 1
                    pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子
                    # print('失败')

            failure_rate = failure_count / (failure_count + success_count)
            # print(failure_rate)

            '--'
            for task in ind:
                task[2].have_been_assign = None
                task[2].commit_T_max_i = None
            for sev in self.sev_group:
                sev.be_use = None

            return failure_rate

    def calculate_pop_method5(self, t_limit):
        """执行短作业优先策略"""
        pop_ind_value = []  # 粒子群内个体的评价值

        for ind in self.pop_group:
            pop_total_t = []
            calc_model = Calculate_Model()  # 计算模型实例
            com_model = Communication_Model()  # 通信模型实例

            failure_count, success_count = 0, 0
            for be_assign_task in ind:  # 将所有任务分配
                '找到任务-最短作业时间'
                max_C_i_value = 100
                min_task_index = None
                index = 0
                for task in ind:  # 找到未分配任务中动态优先级最高的任务
                    task_object = task[2]
                    '比较作业时间'

                    if (task_object.C_i < max_C_i_value) and (task_object.have_been_assign is None):  # 优先级更高且没有分配
                        max_C_i_value = task_object.C_i
                        min_task_index = index
                    index += 1
                task = ind[min_task_index][2]  # 剩余任务中优先级最高的任务
                task.T_max_i = t_limit
                task.have_been_assign = 1  # 被分配

                if task.task_in_local == 0:  # 在本地计算
                    t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                    # print('本地计算的总延时为：{}'.format(t_total))
                    task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                else:
                    '找到服务车'
                    min_t_total = 10  # 服务车中的最小时延
                    index = 0
                    min_index = None
                    for sev in self.sev_group:  # 从服务车中找到时延最小的车
                        if sev.be_use is not None:
                            # print('选择服务车: '.format(direction))
                            dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                            t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                            t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                            if t_total <= task.T_max_i:
                                task.commit_T_max_i = 0
                                if t_total < min_t_total:
                                    min_t_total = t_total  # 更新找到的最小时延的服务车
                                    min_index = index  # 最小车的索引
                            else:
                                task.commit_T_max_i = 1  # 不满足时延限制
                        index += 1

                    '基站计算'
                    if min_index is None:
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                    else:
                        'sev服务'
                        sev = self.sev_group[min_index]
                        sev.be_use = 0  # 被使用
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制

                if task.commit_T_max_i == 0:
                    pop_total_t.append(t_total)
                    success_count += 1
                else:
                    failure_count += 1
                    pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

            failure_rate = failure_count / (failure_count + success_count)
            # print(failure_rate)

            '--'
            for task in ind:
                task[2].have_been_assign = None
                task[2].commit_T_max_i = None
            for sev in self.sev_group:
                sev.be_use = None

            return failure_rate

    def calculate_pop_method6(self, t_limit):

        pop_ind_value = []  # 粒子群内个体的评价值

        for ind in self.pop_group:
            pop_total_t = []
            calc_model = Calculate_Model()  # 计算模型实例
            com_model = Communication_Model()  # 通信模型实例

            failure_count, success_count = 0, 0
            for be_assign_task in ind:  # 将所有任务分配
                '找到任务-时延最小'
                max_C_i_value = 100
                min_task_index = None
                index = 0
                for task in ind:  # 找到未分配任务中动态优先级最高的任务
                    task_object = task[2]
                    '比较作业时间'

                    if (task_object.C_i < max_C_i_value) and (task_object.have_been_assign is None):  # 优先级更高且没有分配
                        max_C_i_value = task_object.C_i
                        min_task_index = index
                    index += 1
                task = ind[min_task_index][2]  # 剩余任务中优先级最高的任务
                task.T_max_i = t_limit
                task.have_been_assign = 1  # 被分配

                if task.task_in_local == 0:  # 在本地计算
                    t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                    # print('本地计算的总延时为：{}'.format(t_total))
                    task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                else:
                    '找到服务车'
                    min_t_total = 10  # 服务车中的最小时延
                    index = 0
                    min_index = None
                    for sev in self.sev_group:  # 从服务车中找到时延最小的车
                        if sev.be_use is not None:
                            # print('选择服务车: '.format(direction))
                            dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                            t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                            t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                            if t_total <= task.T_max_i:
                                task.commit_T_max_i = 0
                                if t_total < min_t_total:
                                    min_t_total = t_total  # 更新找到的最小时延的服务车
                                    min_index = index  # 最小车的索引
                            else:
                                task.commit_T_max_i = 1  # 不满足时延限制
                        index += 1

                    '基站计算'
                    if min_index is None:
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                    else:
                        'sev服务'
                        sev = self.sev_group[min_index]
                        sev.be_use = 0  # 被使用
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制

                if task.commit_T_max_i == 0:
                    pop_total_t.append(t_total)
                    success_count += 1
                else:
                    failure_count += 1
                    pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

            failure_rate = failure_count / (failure_count + success_count)
            # print(failure_rate)

            '--'
            for task in ind:
                task[2].have_been_assign = None
                task[2].commit_T_max_i = None
            for sev in self.sev_group:
                sev.be_use = None

            return failure_rate

    def ini_pop(self, sev_group, bs, task_group):
        """初始化粒子群"""

        # for i in range(0, CONSTANT_SEV):
        #     sev = SeV()  #
        #     self.sev_group.append(sev)  # 添加服务车
        self.sev_group = sev_group
        self.pop_best_ind = []
        self.best_ind_value = 1000
        self.best_ind = None
        # self.bs = BS()  # 创建服务基站实例
        self.bs = bs
        for ind in range(1, CONSTANT_POP_size):  # 构建粒子群
            ind = []  #
            for task in range(1, CONSTANT_TASKS):  # 产生30任务构成一个个体（粒子）
                # taV = TaV()  # 创建任务实例
                taV = task_group[task - 1]

                cm = Calculate_Model()  # 创建计算模型
                t = cm.local_calculate(taV.C_i, taV.max_hash_rate)  # 计算任务需要的计算时间
                acm = Active_Pri_Model()  # 创建动态优先级模型
                taV.pri_i = acm.calculate_pri_i(taV.T_max_i, t, taV.t_stay_i)  # 得到任务的优先级
                v_i = np.random.randint(1, 100, size=(1, CONSTANT_SEV))[0]  # 速度初始坐标
                p_i = np.zeros(CONSTANT_SEV)  # 初始位置：即选择的服务
                num = np.random.randint(1, CONSTANT_SEV, 1)[0]
                p_i[num] = 1
                ind.append([v_i, p_i, taV])  # 单个粒子包括速度和位置两个属性，即两个list, list长度=服务车辆数+基站数
            self.pop_group.append(ind)
        all_ = [self.pop_group, self.sev_group, self.bs]
        # self.print_all(all_)  # 打印初始化内容
        # self.find_all_calculate_self()
        # print('初始化结束')

        # TODO 计算任务的初始优先级

    @staticmethod
    def print_all(all_):
        for i in all_:
            print(i)

    def calculate_pop(self):
        """评估粒子种群"""
        # print('评价粒子群')
        pop_ind_value = []  # 粒子群内个体的评价值
        failure_rate_list = []
        for ind in self.pop_group:
            pop_total_t = []
            calc_model = Calculate_Model()  # 计算模型实例
            com_model = Communication_Model()  # 通信模型实例

            failure_n, n_success = 0, 0
            for task in ind:  # 逐个评价粒子的评价值
                max_index = np.where(task[0] == np.amax(task[0]))  # 方向最大的位置
                direction = max_index[0][0]  # 根据速度最大值选择方向

                task = task[2]  # 待计算的任务
                if self.method == 'exp1_pic3_alg1':
                    '执行自己的算法'
                    if task.task_in_local == 0:  # 在本地计算
                        t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                        # print('本地计算的总延时为：{}'.format(t_total))
                        task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    elif direction == 0:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))
                    else:  # 映射到服务车
                        sev = self.sev_group[direction - 1]  # 选择的sev
                        # print('选择服务车: '.format(direction))
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择SeV的总延时为：{}'.format(t_total))
                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

                elif self.method == 'exp1_pic3_alg2':
                    '执行本地+bs方法'
                    d = random.randint(1, 2)
                    if d == 1:  # 在本地计算
                        t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                        # print('本地计算的总延时为：{}'.format(t_total))
                        task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                    else:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))

                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(t_total + pow(10, 5))  # 不满足时延限制，不可行粒子

                    # print('local+bs：{}', t_total)
                elif self.method == 'exp1_pic3_alg3':
                    '执行随机方法'
                    n_rand = random.randint(1, 3)  # 随机选择一种方法
                    if task.task_in_local == 0:  # 在本地计算
                        if task.task_in_local == 0:  # 可在本地
                            t_total = calc_model.local_calculate(task.C_i, self.bs.f_i)  # 总延时
                            # print('本地计算的总延时为：{}'.format(t_total))
                            task.commit_T_max_i = 0  # 能在本地计算均能满足时延，毋庸置疑，在本地不能计算，传输到其他地方延时更大
                        else:
                            task.commit_T_max_i = 1  # 不能满足时延限制
                    elif n_rand == 2:  # 映射到基站
                        # print('选择基站: '.format(direction))
                        dis = task.calculate_d_i(t=0)  # 计算与bs的距离
                        t_up_ves = com_model.V2I_model(self.bs.g_i, dis, task.D_i)  # 通信时间
                        t_total = calc_model.VEC_calculate(task.C_i, self.bs.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择基站的总延时为：{}'.format(t_total))
                    else:  # 映射到服务车
                        sev = self.sev_group[random.randint(0, CONSTANT_SEV - 1)]  # 随机选择的sev
                        # print('选择服务车: '.format(direction))
                        dis = task.calculate_d_i_j(0, sev.x, sev.vel_j)  # 计算与sev的距离
                        t_up_ves = com_model.V2V_model(sev.g_i_j, dis, task.D_i)  # 通信时间
                        t_total = calc_model.car_calculate(task.C_i, sev.f_i, t_up_ves)  # 总延时
                        if t_total <= task.T_max_i:
                            task.commit_T_max_i = 0
                        else:
                            task.commit_T_max_i = 1  # 不满足时延限制
                        # print('选择SeV的总延时为：{}'.format(t_total))
                    if task.commit_T_max_i == 0:
                        pop_total_t.append(t_total)
                    else:
                        pop_total_t.append(pow(10, 5))  # 不满足时延限制，不可行粒子

                if task.commit_T_max_i == 0:
                    n_success += 1
                else:
                    failure_n += 1

            failure_rate = failure_n / (failure_n + n_success)

            failure_rate_list.append(failure_rate)

            for task in ind:
                task[2].have_been_assign = None
                task[2].commit_T_max_i = None
            for sev in self.sev_group:
                sev.be_use = None

            fit = 0
            # for p in pop_total_t:  # 计算适应值
            #     fit += 1 / p
            fit = np.mean(pop_total_t)  # 适应值改为平均总延时  （根据绘图需求，上面是原文档的适应值公式）

            pop_ind_value.append(fit)

        mean_failure = np.mean(failure_rate_list)

        if self.exp == '2':
            self.pop_best_ind_value = mean_failure
            return mean_failure

        pop_ind_value.sort(reverse=False)
        # print(pop_ind_value)
        min_ind_index = np.where(pop_ind_value == np.amin(pop_ind_value))
        min_ind_index = min_ind_index[0][0]
        # print('评价值最高个体为：{}'.format(max_ind_index))
        # print(self.pop_group[max_ind_index])
        self.pop_best_ind = self.pop_group[min_ind_index]  # 种群最优粒子
        self.pop_best_ind_value = np.amin(pop_ind_value)  # 种群最优粒子评价值
        # print('---------------评估粒子完毕----------------')

    def update_pop(self):
        """更新种群内粒子的位置和速度"""

        for ind in self.pop_group:  # 逐个更新
            count_task = 0
            for task in ind:  # 更新速度和位置
                v1, v2, p1, p2 = 0, 0, 0, 0
                for n_sev in range(len(task[0])):  # 逐个更新每一个位置的速度
                    v1, p1 = task[0][n_sev], task[1][n_sev]

                    task[0][n_sev] = self.w * task[0][n_sev] \
                                     + self.c1 * np.random.random() * (
                                             self.pop_best_ind[count_task][1][n_sev] - task[1][n_sev]) \
                                     + self.c2 * np.random.random() * (
                                             self.best_ind[count_task][1][n_sev] - task[1][n_sev])  # 更新v
                    task[1][n_sev] += task[0][n_sev]  # 更新p
                    v2, p2 = task[0][n_sev], task[1][n_sev]
                # print('v1,p1:{0}{1}-->v2,p2:{2}{3}'.format(v1, p1, v2, p2))  # 打印例子粒子更新过程
                count_task += 1

    def __update_pop__(self):
        """更新种群内粒子的位置和速度"""

        for ind in self.pop_group:  # 逐个更新
            count_task = 0
            for task in ind:  # 更新速度和位置
                v1, v2, p1, p2 = 0, 0, 0, 0
                for n_sev in range(len(task[0])):  # 逐个更新每一个位置的速度
                    v1, p1 = task[0][n_sev], task[1][n_sev]

                    task[0][n_sev] = random.uniform(0, 2) * task[0][n_sev]  # 更新v
                    task[1][n_sev] += random.uniform(0, 1)  # 更新p
                    v2, p2 = task[0][n_sev], task[1][n_sev]
                # print('v1,p1:{0}{1}-->v2,p2:{2}{3}'.format(v1, p1, v2, p2))  # 打印例子粒子更新过程
                count_task += 1

    def update_pop_best_position(self):
        """更新种群的全局最优位置"""
        if self.method == 'exp1_pic3_alg2':
            print('')
            pass
        if self.pop_best_ind_value < self.best_ind_value:
            self.best_ind = self.pop_best_ind  # 更新最优粒子
            self.best_ind_value = self.pop_best_ind_value
            if self.pop_best_ind_value < 2:
                self.his_best_value.append(self.pop_best_ind_value)  # 记录历史最优结果，画图用
                self.his_best_value_time.append(time.time() - self.begin_time)  # 记录获得该值得时间，画图用
                # print('更新最优粒子的评价值更新为：{}'.format(self.best_ind_value))
