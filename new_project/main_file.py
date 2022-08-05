import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

import algs
from matplotlib.font_manager import FontProperties #字体管理器
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)


class Fleet:
    """任务队列"""

    def __init__(self, number_task, index=None):
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
        self.adj = index
        self.res = None
        # self.adj_ = None

    def create_position(self):
        for n in self.ini_list:
            self.position[n] = (random.randint(-500, 500), 0)  # 任务位置
            self.toward[n] = random.choice([-1, 1])  # (左，右)

    def clear(self, res=None):
        self.task_wait_t = {}  # 等待时间
        self.task_calcu_t = {}  # 计算时间
        self.task_finish_t = {}  # 总完成时长
        # if res is not None self.res = sorted(res, )

    def pri_j(self):
        # 计算优先级的三个重要参数
        for task in range(self.number):
            T_MAX_i = random.uniform(0, 3)  # [0, 3]s
            C_i = random.uniform(1, 3)  # [1, 3]GHz
            import_i = random.randint(1, 3)  # [1,2,3]
            self.pri_j_task_fleet_const[task] = [T_MAX_i, C_i, import_i]

    def clear_(self, res=None):
        self.task_wait_t = {}  # 等待时间
        self.task_calcu_t = {}  # 计算时间
        self.task_finish_t = {}  # 总完成时长
        if res is not None: self.res = sorted(res, )


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

    def pic1(self, plt):
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
                label = 'comp1'
                marker = '*'
            else:
                label = 'comp2'
                marker = '*'
            index += 1
            plt.plot(plt_res[0], plt_res[1], label=label, marker=marker)
        plt.set_title(label='pic1')
        plt.set_xlabel(xlabel='服务器的计算能力/f', fontproperties=font)
        plt.set_ylabel(ylabel='任务失败率/百分比', fontproperties=font)
        # plt.set_rcParams['font.sans-serif'] = [u'SimHei']
        # plt.set_rcParams['axes.unicode_minus'] = False
        plt.legend()
        # plt.show()
        # plt.clf()
        return plt


    def pic2(self, plt):
        """my_alg vs alg1 and alg2"""
        # 自变量  ves最大计算能力
        T_MAX_i_list = np.arange(0.1, 3, 0.3)

        # 固定参数
        max_volume_vec_f = 10
        break_list = np.arange(0, 100, 0.5)  # ves判断抢占的时间点
        n_task = 30
        para_abv = para_abv_func()
        my_res, alg1_res, alg2_res = [[], []], [[], []], [[], []]
        for T_MAX_i in list(T_MAX_i_list):
            # max_volume_vec_f = v
            # print(self.type_pic[0])
            fleet = Fleet(n_task)
            for i in fleet.ini_list: fleet.pri_j_task_fleet_const[i][0] = random.uniform(0, T_MAX_i)

            res = algs.my_alg_func(task_fleet=fleet, n_pic='pic2', n_task=fleet.number,
                                   volume_vec_f=max_volume_vec_f, T_MAX_i=T_MAX_i, t_dev=break_list,
                                   para_abv=para_abv)  # [[], []]
            my_res[0].append(T_MAX_i)
            my_res[1].append(res)
            fleet.clear()

            res = algs.alg1_func(task_fleet=fleet, n_pic='pic2', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=T_MAX_i, t_dev=0.1,
                                 para_abv=para_abv)  # [[], []]
            alg1_res[0].append(T_MAX_i)
            alg1_res[1].append(res)
            fleet.clear()

            res = algs.alg2_func(task_fleet=fleet, n_pic='pic2', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=0.1,
                                 para_abv=para_abv)  # [[], []]
            alg2_res[0].append(T_MAX_i)
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
                label = 'comp1'
                marker = '*'
            else:
                label = 'comp2'
                marker = '*'
            index += 1
            plt.plot(plt_res[0], plt_res[1], label=label, marker=marker)
        plt.set_title(label='pic2')
        plt.set_xlabel(xlabel='任务的最大时延/s',fontproperties=font)
        plt.set_ylabel(ylabel='任务失败率/百分比',fontproperties=font)
        # plt.rcParams['font.sans-serif'] = [u'SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        plt.legend()
        # plt.clf()
        return plt

    def pic3(self, plt):
        """my_alg vs alg1 and alg2"""
        # 自变量  ves最大计算能力
        n_task_list = np.arange(30, 100, 1)  # 范围
        # 固定参数
        break_list = np.arange(0, 100, 0.5)  # ves判断抢占的时间点
        # n_task = 30
        para_abv = para_abv_func()
        my_res, alg1_res, alg2_res = [[], []], [[], []], [[], []]
        for n_task in list(n_task_list):
            max_volume_vec_f = 10
            # print(self.type_pic[0])
            fleet = Fleet(n_task)
            res = algs.my_alg_func(task_fleet=fleet, n_pic='pic3', n_task=fleet.number,
                                   volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=break_list,
                                   para_abv=para_abv)  # [[], []]
            my_res[0].append(n_task)
            my_res[1].append(res)
            fleet.clear()

            res = algs.alg1_func(task_fleet=fleet, n_pic='pic3', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=0.1,
                                 para_abv=para_abv)  # [[], []]
            alg1_res[0].append(n_task)
            alg1_res[1].append(res)
            fleet.clear()

            res = algs.alg2_func(task_fleet=fleet, n_pic='pic3', n_task=fleet.number,
                                 volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=0.1,
                                 para_abv=para_abv)  # [[], []]
            alg2_res[0].append(n_task)
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
                label = 'comp1'
                marker = '*'
            else:
                label = 'comp2'
                marker = '*'
            index += 1
            plt.plot(plt_res[0], plt_res[1], label=label, marker=marker)
        plt.set_title(label='pic3')
        plt.set_xlabel(xlabel='车辆数目',fontproperties=font)
        plt.set_ylabel(ylabel='任务失败率/百分比',fontproperties=font)
        # plt.rcParams['font.sans-serif'] = [u'SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        plt.legend()
        # plt.clf()
        # plt.show()
        return plt


    def pic4(self, plt):
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
            # print('--my--')
            res = algs.my_alg_func(task_fleet=fleet, n_pic='pic4', n_task=fleet.number,
                                   volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=t_dev,
                                   para_abv=para_abv)  # [[], []]
            my_res[0].append(n_task)
            my_res[1].append(res)
            fleet.clear()
            # print('--alg3--')
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
                label = 'comp3'
                marker = '*'
            index += 1
            plt.plot(plt_res[0], plt_res[1], label=label, marker=marker)
        plt.set_title(label='pic4')
        plt.set_xlabel(xlabel='车辆数',fontproperties=font)
        plt.set_ylabel(ylabel='任务切换次数',fontproperties=font)
        plt.legend()
        # plt.rcParams['font.sans-serif'] = [u'SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.show()
        return plt

    def pic5(self, plt):

        print('--pic5--')
        # 自变量  任务数量
        task_number_list = np.arange(30, 100, 1)  # 范围
        dev_list = np.arange(0.4, 3, 0.1)
        dev_list_list = []
        for dev_ in dev_list:
            list_ = np.arange(0.1, 500000, dev_)
            dev_list_list.append(list_)  # 间隔0.3检测
        # 固定参数
        n_task = 100
        max_volume_vec_f = 10  # ves最大计算能力
        para_abv = para_abv_func()
        my_res, alg3_res = [[], []], [[], []]
        index_dev = 0
        res = []
        fleet = Fleet(n_task, 0)
        for t_dev in list(dev_list_list):
            fleet = Fleet(n_task, index_dev)  # 生成任务集
            # print(self.type_pic[0])
            # print('--my--')
            res.append(algs.my_alg_pic5(task_fleet=fleet, n_pic='pic5', n_task=fleet.number,
                                   volume_vec_f=max_volume_vec_f, T_MAX_i=None, t_dev=t_dev,
                                   para_abv=para_abv))  # [[], []]
            index_dev += 1
        # else:
            # fleet, res = Fleet(n_task, index_dev)  # 生成任务集
        fleet.clear_(res)
        my_res[0] = dev_list
        my_res[1] = fleet.res

        '绘图'
        label = 'my_alg'
        marker = '^'
        plt.plot(my_res[0], my_res[1], label=label, marker=marker)
        plt.set_title(label='pic5')
        plt.set_xlabel(xlabel='检测时间间隔/s',fontproperties=font)
        plt.set_ylabel(ylabel='任务失败率',fontproperties=font)
        # plt.set_rcParams['font.sans-serif'] = [u'SimHei']
        # plt.set_rcParams['axes.unicode_minus'] = False
        plt.legend()
        # plt.show()
        return plt


if __name__ == "__main__":
    # 5张图
    pic = Picture()
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()


    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax4 = fig4.add_subplot(1, 1, 1)
    ax5 = fig5.add_subplot(1, 1, 1)


    ax1 = pic.pic1(ax1)
    ax2 = pic.pic2(ax2)
    ax3 = pic.pic3(ax3)
    ax4 = pic.pic4(ax4)
    ax5 = pic.pic5(ax5)
    plt.show()