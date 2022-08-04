import random

import matplotlib.pyplot as plt
import numpy as np

import algs


class Fleet:
    """任务队列"""

    def __init__(self, number_task):
        self.number = number_task
        self.ini_list = np.arange(0, self.number, 1)  # 初始队列排序
        self.pri_j_task_fleet_const = {}

        self.task_wait_t = {}  # 等待时间
        self.task_calcu_t = {}  # 计算时间
        self.task_finish_t = {}  # 总完成时长
        self.task_wait_t_before_assign = {}  # 分配之前的等待时间
        self.task_calcu_t_before_assign = {}  # 分配之前的等待时间
        self.rest_use_t = {}  # 车辆距离边界的离开时长

        self.position = {}  # 位置
        self.toward = {}  # 方向

        self.create_position()
        self.pri_j()

    def create_position(self):
        for n in self.ini_list:
            self.position[n] = (random.randint(-500, 500), 0)  # 任务位置
            self.toward[n] = random.choice([-1, 1])  # (左，右)

    def clear(self):
        self.task_wait_t = {}  # 等待时间
        self.task_calcu_t = {}  # 计算时间
        self.task_finish_t = {}  # 总完成时长

    def pri_j(self):
        # 计算优先级的三个重要参数
        for task in range(self.number):
            T_MAX_i = random.uniform(0, 3)  # [0, 3]s
            C_i = random.uniform(1, 3)  # [1, 3]GHz
            import_i = random.randint(1, 3)  # [1,2,3]
            self.pri_j_task_fleet_const[task] = [T_MAX_i, C_i, import_i]

    # def task_time(self):
    #     for n in self.ini_list:
    #         self.task_wait_t[n] = 0
    #         self.task_calcu_t = -1
    #         self.task_finish_t = -1


def para_abv_func():
    a = random.uniform(0, 0.5)
    b = random.uniform(0, 0.5)
    v = 1 - a - b
    para = {'a': a, 'b': b, 'v': v}
    return para


class Picture:

    def __int__(self):
        self.num_pic = 5
        self.type_pic = ['折线', '柱状']
        self.alg_name = ['my_alg', 'alg1_静态优先级', 'alg2_动态最高优先数优先', 'alg3_直接抢占']

    def pic1(self):
        """my_alg vs alg1 and alg2"""
        # 自变量  ves最大计算能力
        v_volume_list = np.arange(5, 10, 0.5)  # 范围
        # 固定参数
        break_list = np.arange(0, 100, 0.5)  # ves判断抢占的时间点
        n_task = 30
        para_abv = para_abv_func()
        my_res, alg1_res, alg2_res = [[], []], [[], []], [[], []]
        for v in list(v_volume_list):
            max_volume_vec_f = v
            # print(self.type_pic[0])
            fleet = Fleet(n_task)
            res = algs.my_alg_func(task_fleet=fleet, n_pic='pic1', n_task=fleet.number,
                                   volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=break_list,
                                   para_abv=para_abv)  # [[], []]
            my_res[0].append(v)
            my_res[1].append(res)
            fleet.clear()

            res = algs.alg1_func(task_fleet=fleet, n_pic='pic1', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=0.1,
                                 para_abv=para_abv)  # [[], []]
            alg1_res[0].append(v)
            alg1_res[1].append(res)
            fleet.clear()

            res = algs.alg2_func(task_fleet=fleet, n_pic='pic1', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=0.1,
                                 para_abv=para_abv)  # [[], []]
            alg2_res[0].append(v)
            alg2_res[1].append(res)
            fleet.clear()
        else:
            print('结束')

        '绘图'
        index = 0
        for plt_res in [my_res, alg1_res, alg2_res]:
            if index == 0:
                label = 'my_alg'
                marker = '^'
            elif index == 1:
                label = '对比1'
                marker = '*'
            else:
                label = '对比2'
                marker = '*'
            index += 1
            plt.plot(plt_res[0], plt_res[1], label=label, marker=marker)
        plt.xlabel(xlabel='服务器的计算能力/f')
        plt.ylabel(ylabel='任务失败率/百分比')
        plt.rcParams['font.sans-serif'] = [u'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend()
        plt.show()
        plt.clf()

    def pic4(self):
        """my_alg vs alg3"""
        print('--pic4--')
        # 自变量  任务数量
        task_number_list = np.arange(30, 100, 1)  # 范围
        t_dev = np.arange(0.1, 500000, 0.5)  # 间隔0.3检测
        # 固定参数
        max_volume_vec_f = 10  # ves最大计算能力
        para_abv = para_abv_func()
        my_res, alg3_res = [[], []], [[], []]
        for n_task in list(task_number_list):
            # print(self.type_pic[0])
            fleet = Fleet(n_task)  # 生成任务集
            print('--my--')
            res = algs.my_alg_func(task_fleet=fleet, n_pic='pic4', n_task=fleet.number,
                                   volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=t_dev,
                                   para_abv=para_abv)  # [[], []]
            my_res[0].append(n_task)
            my_res[1].append(res)
            fleet.clear()
            print('--alg3--')
            res = algs.alg3_func(task_fleet=fleet, n_pic='pic4', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=t_dev,
                                 para_abv=para_abv)  # [[], []]
            alg3_res[0].append(n_task)
            alg3_res[1].append(res)
            fleet.clear()

        else:
            print('结束')

        '绘图'
        index = 0
        for plt_res in [my_res, alg3_res]:
            if index == 0:
                label = 'my_alg'
                marker = '^'
            else:
                label = '对比3'
                marker = '*'
            index += 1
            plt.plot(plt_res[0], plt_res[1], label=label, marker=marker)
        plt.xlabel(xlabel='车辆数')
        plt.ylabel(ylabel='任务切换次数', rotation=1)
        plt.rcParams['font.sans-serif'] = [u'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # 5张图
    pic = Picture()
    pic.pic1()
    pic.pic4()
