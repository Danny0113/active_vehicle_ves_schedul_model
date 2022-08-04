from class_tools import *

if __name__ == '__main__':
    comp_exp = Compare_Exp(exp='1', CONSTANT_ITER=100)  # 创建实验实例
    '执行两个对比实验'
    # comp_exp.exp1_pic3()  # 执行对比实验1
    # comp_exp.exp2_pic2()  # 执行对比实验2

    '以下单独执行本文算法'
    pso = PSO(exp='1', CONSTANT_ITER=100)
    sev_group, bs, task_group = comp_exp.create_case()  # 生成数据集
    res1 = pso.run(sev_group, bs, task_group, method='exp1_pic3_alg1')  # 执行本文算法
