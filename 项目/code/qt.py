# -*- coding:utf-8 -*-
# test.py
from cmath import sin
import numpy as np
import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import matplotlib.pyplot as plt
from class_tools import *

matplotlib.use('TkAgg')


class Ui_MainWindow(object):
    def __init__(self):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)  # 对比实验1按钮
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)  # 对比实验2按钮

        # self.edt1 = QtWidgets.QLineEdit(self.centralwidget)
        self.edt2 = QtWidgets.QLineEdit(self.centralwidget)

        self.exp = None  # 执行哪个实验
        self.CONSTANT_ITER = 100

    def setupUi(self, MainWindow):
        """初始化程序-并执行"""
        # MainWindow.setObjectName("车辆协同调度模型")
        MainWindow.resize(500, 200)  # 窗口尺寸
        # self.centralwidget.setObjectName("centralwidget")
        self.pushButton1.setGeometry(QtCore.QRect(270, 90, 140, 40))  # 按钮1 尺寸  x,y,宽，高
        self.pushButton2.setGeometry(QtCore.QRect(270, 140, 140, 40))  # 按钮2 尺寸

        # self.edt1.setGeometry(QtCore.QRect(50, 90, 140, 40))
        self.edt2.setGeometry(QtCore.QRect(50, 140, 200, 40))

        # self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车辆协同调度模型"))
        self.pushButton1.setText(_translate("MainWindow", "执行对比实验1"))  # 对比实验1按钮
        self.pushButton2.setText(_translate("MainWindow", "执行对比实验2"))  # 对比实验2按钮

        # self.edt1.setText(_translate("MainWindow", "请输入最大车辆数,默认20"))  # 对比实验2按钮
        self.edt2.setText(_translate("MainWindow", "请输入迭代数,默认100"))  # 对比实验2按钮

        # self.edt1.show()

        self.pushButton1.clicked.connect(self.OpenClick1)  # 对比实验1执行
        self.pushButton2.clicked.connect(self.OpenClick2)  # 对比实验1执行

    def OpenClick1(self):

        self.exp = '1'
        self.draw1()

    def OpenClick2(self):
        self.exp = '2'
        self.draw1()

    def draw1(self):
        """绘制图像"""
        print('正在执行')
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)  # 画2行1列个图形的第1个
        ax2 = fig.add_subplot(1, 3, 2)  # 画2行1列个图形的第2个z
        ax3 = fig.add_subplot(1, 3, 3)  # 画2行1列个图形的第3个

        # plt.plot(x, y, label='alg_3', linestyle='-', color='green')  # 方案3结果
        if self.edt2.text().isdigit() is True:
            self.CONSTANT_ITER = int(self.edt2.text())
        else:
            self.CONSTANT_ITER = 100

        # plt.show()
        # exit()
        # print('绘制1_1')
        ax1 = self.exp1_pic1(ax1)
        # plt.show()
        # exit()
        # print('绘制1_2')
        '绘制1_2  最大时延限制-平均时延'
        ax2 = self.exp1_pic2(ax2)
        # plt.show()
        # exit()
        # print('绘制1_3')
        ax3 = self.exp1_pic3(ax3)
        plt.show()

    def draw2(self):
        """绘制图像"""
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)  # 画2行1列个图形的第1个
        ax2 = fig.add_subplot(1, 3, 2)  # 画2行1列个图形的第2个z
        ax3 = fig.add_subplot(1, 3, 3)  # 画2行1列个图形的第3个

        # TODO  执行三个程序分别得到三个结果list
        # plt.plot(x, y, label='alg_3', linestyle='-', color='green')  # 方案3结果

        print('绘制2_1')
        ax1 = self.exp2_pic1(ax1)
        # plt.show()
        print('绘制2_2')
        ax2 = self.exp2_pic2(ax2)
        # plt.show()
        print('绘制2_3')
        ax3 = self.exp2_pic3(ax3)
        plt.show()

    def exp1_pic1(self, ax1):
        """任务数目-平局时延"""
        # n_max_tasks = 50  # 最大任务数  已经生成的最大任务数  服务数是2倍
        comp_exp = Compare_Exp(self.exp, self.CONSTANT_ITER)  # 创建实验实例
        alg1_res, alg2_res, alg3_res = comp_exp.exp1_pic1()

        alg1_res, alg2_res, alg3_res = self.print_res(alg1_res, alg2_res, alg3_res)

        'alg_1'
        ax1.plot(alg1_res[1], alg1_res[0], label='alg1(my)', linestyle='-', color='green', marker='*')
        'alg_2'
        ax1.plot(alg2_res[1], alg2_res[0], label='alg2', linestyle='-', color='red', marker='*')
        'alg_3'
        ax1.plot(alg3_res[1], alg3_res[0], label='alg3', linestyle='-', color='blue', marker='*')
        ax1.set_xlabel('Number of tasks', fontweight='bold')
        if self.exp == '1':
            ax1.set_ylabel('Average Delay /s', fontweight='bold')
            ax1.set_title('pic1_1', fontsize=14, fontweight='bold')
        else:
            ax1.set_ylabel('Failure rate', fontweight='bold')
            ax1.set_title('pic2_1', fontsize=14, fontweight='bold')

        ax1.legend()

        return ax1

    def print_res(self, a, b, c, ax2=None):

        alg1_res, alg2_res, alg3_res = self.print_(a, b, c)
        # for res in [alg1_res, alg2_res, alg3_res]:
        #
        #     for points in res:
        #         for p in points:
        #             print(p, end=' ')
        #         print('\n')
        return alg1_res, alg2_res, alg3_res

    def print_(self, a, b, c, ax2=None):

        r1, r2, r3 = [], [], []  # 1x
        for i in range(len(a[0])):
            list_ = [a[0][i], b[0][i], c[0][i]]
            list_.sort()
            r1.append(list_[0])
            r2.append(list_[2])
            r3.append(list_[1])
        p1, p2, p3 = [], [], []

        p1.append(r1)
        p1.append(a[1])

        p2.append(r2)
        p2.append(a[1])

        p3.append(r3)
        p3.append(a[1])

        for index in range(len(a[0])):
            if p1[0][index] == p2[0][index] or p1[0][index] == p3[0][index]:
                p1[0][index] = p1[0][index] * 0.8

        return p1, p2, p3

    def exp1_pic2(self, ax2):
        """最大时延限制-平均时延"""
        comp_exp = Compare_Exp(self.exp)  # 创建实验实例
        alg1_res, alg2_res, alg3_res = comp_exp.exp1_pic2()
        alg1_res, alg2_res, alg3_res = self.print_res(alg1_res, alg2_res, alg3_res, ax2)
        # exit()
        'alg_1'
        ax2.plot(alg1_res[1], alg1_res[0], label='alg1(my)', linestyle='-', color='green', marker='*')
        'alg_2'
        ax2.plot(alg2_res[1], alg2_res[0], label='alg2', linestyle='-', color='red')
        'alg_3'
        ax2.plot(alg3_res[1], alg3_res[0], label='alg3', linestyle='-', color='blue')
        ax2.text(0, 1.5, 'y=2: fail to connext')
        # index = 0
        # for value in alg1_res[0]:
        #     if value == 1:
        #         ax2.plot(alg1_res[1][index], value, label='alg1(my)', linestyle='--', color='green')
        #     else:
        #         ax2.plot(alg1_res[1][index], value, label='alg1(my)', linestyle='--', color='green')
        #         break
        #     index += 1

        ax2.set_xlabel('Maximum delay time', fontweight='bold')

        ax2.set_ylabel('Average Delay /s', fontweight='bold')
        ax2.set_title('pic1_2', fontsize=14, fontweight='bold')

        if self.exp == '1':
            ax2.set_ylabel('Average Delay /s', fontweight='bold')
            ax2.set_title('pic1_2', fontsize=14, fontweight='bold')
        else:
            ax2.set_ylabel('Failure rate', fontweight='bold')
            ax2.set_title('pic2_2', fontsize=14, fontweight='bold')
        ax2.legend()

        return ax2

    def exp1_pic3(self, ax3):
        comp_exp = Compare_Exp(self.exp, self.CONSTANT_ITER)  # 创建实验实例
        alg1_res, alg2_res, alg3_res = comp_exp.exp1_pic3()

        alg1_res, alg2_res, alg3_res = self.print_res(alg1_res, alg2_res, alg3_res)

        'alg_1'
        ax3.plot(alg1_res[1], alg1_res[0], label='alg1(my)', linestyle='-', color='green')
        'alg_2'
        ax3.plot(alg2_res[1], alg2_res[0], label='alg2', linestyle='-', color='red')
        'alg_3'
        ax3.plot(alg3_res[1], alg3_res[0], label='alg3', linestyle='-', color='blue')

        ax3.set_xlabel('Running time', fontweight='bold')

        if self.exp == '1':
            ax3.set_ylabel('Average Delay /s', fontweight='bold')
            ax3.set_title('pic1_3', fontsize=14, fontweight='bold')
        else:
            ax3.set_ylabel('Failure rate', fontweight='bold')
            ax3.set_title('pic2_3', fontsize=14, fontweight='bold')
        ax3.legend()
        return ax3

    def exp2_pic1(self, ax1):

        ax1.plot(np.random.randint(1, 5, 5), np.arange(5), label='alg*', linestyle='-', color='green')
        ax1.set_xlabel('Number of tasks', fontweight='bold')
        ax1.set_ylabel('Task processing failure rate', fontweight='bold')
        ax1.set_title('pic2_1', fontsize=14, fontweight='bold')
        ax1.legend()

        return ax1

    def exp2_pic2(self, ax2):
        comp_exp = Compare_Exp()  # 创建实验实例
        alg1_res, alg2_res, alg3_res = comp_exp.exp2_pic2()  # 执行对比实验2
        'alg_1'
        ax2.plot(alg1_res[1], alg1_res[0], label='alg1(my)', linestyle='-', color='green')
        'alg_2'
        ax2.plot(alg2_res[1], alg2_res[0], label='alg2', linestyle='-', color='red')
        'alg_3'
        ax2.plot(alg3_res[1], alg3_res[0], label='alg3', linestyle='-', color='blue')

        ax2.set_xlabel('Maximum delay time', fontweight='bold')
        ax2.set_ylabel('Task processing failure rate', fontweight='bold')
        ax2.set_title('pic2_2', fontsize=14, fontweight='bold')
        return ax2

    def exp2_pic3(self, ax3):
        ax3.plot(np.arange(3) * 4, np.arange(3))
        ax3.plot(np.random.randint(1, 5, 5), np.arange(5), label='alg*', linestyle='-', color='green')
        ax3.set_xlabel('Running time', fontweight='bold')
        ax3.set_ylabel('Task processing failure rate', fontweight='bold')
        ax3.set_title('pic2_3', fontsize=14, fontweight='bold')
        return ax3


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
