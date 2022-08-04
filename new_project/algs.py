"""参数"""

#  路长  1000m
#  基站位置  路中间, 覆盖半径为500m
#  基站 最大计算能力f=10GHz

#  任务车产生独立任务 大小 [100, 300]kb
#  (0)最大时延限制 T_MAX_i [0, 3]s
#  (1)所需资源为 Ci[1, 3]GHz
#  (2)任务重要性因子 import_i = [1,2,3]
#  车速  40km/h


#  车辆与基站之间链路带宽 H = 5MHz
#  信道增益 g_i = 20e-4
#  车辆传输功率 p_i = 100MW
#  噪声功率  Q = 10e-10


import numpy as np
import pandas as pd


class Ves:

    def __init__(self, volume_vec_f):
        self.volume = volume_vec_f  # 最大容量
        self.rest_volume = volume_vec_f  # 当前剩余容量
        self.over_current_task_time = 0  # 完成已分配任务需要的时间
        self.current = 0  # 当前时间
        self.task_rest_t_list = {}  # 任务剩余完成时间长短队列


def calcu_task_pri_in_list(task_fleet_const, task_list, para=dict):
    # 三个优先级因素的系数，之和为1
    if para is None:
        para = {'a': 0, 'b': 0, 'v': 0}

    # pr_j = [0, 0, 0]
    # 导入任务
    task_pri_mati = []
    for n_task in task_list:
        task_pri_mati.append(task_fleet_const[n_task])
    # 标准化
    np_array = np.array(task_pri_mati)
    # 1.指标的最大最小值[x1,x2,x3]
    min_pri_j = np.min(np_array, axis=0)
    max_pri_j = np.max(np_array, axis=0)
    # 2.计算
    for d in [0, 1, 2]:
        np_array[:, d] = (np_array[:, d] - min_pri_j[d]) / (max_pri_j[d] - min_pri_j[d])
    # 计算任务优先级
    task_pri_list = np.zeros(len(task_list))
    for task in task_list:
        task_pri_list[task] = para['a'] * np_array[task, 0] + para['b'] * np_array[task, 1] + para['v'] * np_array[
            task, 2]
    task_pri_dict = {}
    for task in task_list:
        task_pri_dict[task] = task_pri_list[task]
    return task_pri_dict  # {'task':pri_value,...,}


def get_calcu_time(Ci, f_max, t_interrupt_i=0, be_break=False):
    # 任务计算时间
    if not be_break:
        t_com = round(Ci / f_max, 2)
    else:
        t_com = round(Ci / f_max, 2) + t_interrupt_i
        pass
    return t_com


def calcu_failure_rate(fleet, first_failure_list):
    # 计算失败率
    num_failure = 0

    for num in fleet.ini_list:
        if num in first_failure_list:
            num_failure += 1
        elif fleet.pri_j_task_fleet_const[num][0] < fleet.task_finish_t[num]:
            num_failure += 1
    rate = num_failure / fleet.number
    return rate


def clean_task(fleet):
    # 计算使出边界的用时
    list_task_f = []
    for n in fleet.ini_list:
        if fleet.toward[n] == -1:  # 左
            dis = fleet.position[n][0] - (-500)
            use_t = ((dis / 1000) / 40) * 3600  # 秒
            if use_t < fleet.pri_j_task_fleet_const[n][0]:
                list_task_f.append(n)
        else:
            dis = 500 - fleet.position[n][0]
            use_t = ((dis / 1000) / 40) * 3600  # 秒
            if use_t < fleet.pri_j_task_fleet_const[n][0]:
                list_task_f.append(n)
        fleet.rest_use_t[n] = use_t

    return list_task_f, fleet


def update_pri_in_my_alg(fleet, rest_task_list):
    task_pri_dict = {}
    for n_task in rest_task_list:
        task_pri_dict[n_task] = (fleet.task_wait_t_before_assign[n_task] + fleet.task_calcu_t_before_assign[n_task]) / \
                                fleet.task_calcu_t_before_assign[n_task]

    return task_pri_dict


def adj_row_use_index(task_pri_df, task):
    list_index = list(task_pri_df.index)
    list_value = list(task_pri_df[0].values)
    # print(list_index,'\n',list_value)
    index_of_value = list_index.index(task)
    list_index.remove(task)
    list_index.insert(0, task)
    value = list_value[index_of_value]
    del list_value[index_of_value]
    list_value.insert(0, value)
    # print(list_index, '\n', list_value)
    task_pri_df = pd.DataFrame(list_value, index=list_index)

    return task_pri_df


def judge_rest_task(task_fleet, task_pri_df):
    # 只考虑两辆车则，发现满足条件的task就跳出
    task_list = list(task_pri_df.index)
    for task in task_list:
        # 时限【定值】-等待时间【变化】 vs 计算时间【定值】
        t_i_max = task_fleet.pri_j_task_fleet_const[task][0]  # 时限
        t_wait_i = task_fleet.task_wait_t_before_assign[task]  # 等待时间
        t_calcu_i = task_fleet.task_calcu_t_before_assign[task]  # 计算时间
        t_wait_max = round(t_i_max / 2, 3)  # 阈值
        if t_i_max - t_wait_i > t_calcu_i:  # 条件1：task在最大时限剩余时间内是否能完成计算任务
            if t_wait_i > t_wait_max:  # 条件2：任务等待时间是否超过阈值
                # 需要调整优先级
                task_pri_df = adj_row_use_index(task_pri_df, task)
                return task_pri_df, True
            else:
                continue
        else:
            continue

    return task_pri_df, False  # 不满足


def my_alg_func(task_fleet, n_pic, n_task, volume_vec_f, T_MAX_i, t_dev, para_abv):
    """动态调整任务优先级  抢占式调度
        n_pic: 实验类型
        volume_vec_f: pic1
        T_MAX_i: pic2
        n_task: pic3 and 4
        t_dev: pic5
        """

    # 计算不能在最大时延内完成的任务
    task_list_failure, task_fleet = clean_task(fleet=task_fleet)
    # 计算任务的执行时间
    for num_task in task_fleet.ini_list:
        task_fleet.task_calcu_t_before_assign[num_task] = get_calcu_time(
            Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
            f_max=volume_vec_f)
    # 计算任务优先级-->优先级队列  相同则按T_max_i: 小-->大
    task_pri_dict = calcu_task_pri_in_list(task_fleet_const=task_fleet.pri_j_task_fleet_const,
                                           task_list=task_fleet.ini_list, para=para_abv)
    df = pd.DataFrame([task_pri_dict])  # df 好处理
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df2 = df2.sort_values(by=0, ascending=False)  # 按优先级排列
    one_ves = Ves(volume_vec_f)
    # 初始化等待时间
    for num in task_fleet.ini_list:
        task_fleet.task_wait_t_before_assign[num] = 0
    # 完不成的不考虑
    df2 = df2.drop(index=task_list_failure)

    while df2.empty is False:  # 存在未调度的任务

        rest_list = list(df2.index)
        # 更新任务等待时间
        for num in rest_list:
            task_fleet.task_wait_t_before_assign[num] = one_ves.current
        # 计算任务优先级
        task_pri_dict = update_pri_in_my_alg(fleet=task_fleet, rest_task_list=rest_list)
        df = pd.DataFrame([task_pri_dict])  # df 好处理
        df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        df2 = df2.sort_values(by=0, ascending=False)
        # 判断是否存在存在满足调整优先级条件的task
        df2, adj_flag = judge_rest_task(task_fleet=task_fleet, task_pri_df=df2)
        # 优先级最高任务
        num_task = df2.index[0]
        # ves是否立即满足计算需求
        task_volume = task_fleet.pri_j_task_fleet_const[num_task][1]
        # 容量是否满足
        if one_ves.rest_volume >= task_volume:
            # 任务等待时间
            task_fleet.task_wait_t[num_task] = one_ves.current - 0
            # 任务的计算时间
            task_fleet.task_calcu_t[num_task] = get_calcu_time(Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
                                                               f_max=volume_vec_f)
            # 完成时间
            task_fleet.task_finish_t[num_task] = task_fleet.task_wait_t[num_task] + task_fleet.task_calcu_t[num_task]
            task_finish_t = pd.Series(task_fleet.task_finish_t)
            # 更新ves
            # 1.任务剩余完成时间队列
            one_ves.task_rest_t_list[num_task] = task_fleet.task_calcu_t[num_task]
            # 2.更新优先级队列
            df2 = df2.drop(index=num_task)
            # 3.更新ves容量
            one_ves.rest_volume -= task_volume
        else:  # 当前容量不足
            data = pd.Series(one_ves.task_rest_t_list)
            series = data.sort_values(ascending=True)
            min_key = series.keys()[0]
            # 1.更新当前时刻
            one_ves.current += series[min_key]
            # 2.剔除一个任务
            del one_ves.task_rest_t_list[min_key]
            # 3.释放空间
            one_ves.rest_volume += task_fleet.pri_j_task_fleet_const[min_key][1]
            # 4.更新任务剩余完成时间队列
            task_rest_t_list = {}
            for key in one_ves.task_rest_t_list:
                one_ves.task_rest_t_list[key] = one_ves.task_rest_t_list[key] - series[min_key]
                rest_t = one_ves.task_rest_t_list[key]
                task_rest_t = pd.Series(one_ves.task_rest_t_list)

    # 计算失败率
    f_fate = calcu_failure_rate(fleet=task_fleet, first_failure_list=task_list_failure)

    print('任务分配完毕')
    if n_pic in ['pic1', 'pic2', 'pic3']:
        return f_fate  # '任务失败率'
    elif n_pic == 'pic4':
        return '任务切换次数'
    elif n_pic == 'pic5':
        return '任务失败率'


def alg1_func(task_fleet, n_pic, n_task, volume_vec_f, T_MAX_i, t_dev, para_abv):
    """静态任务优先级  非抢占式调度"""
    # 计算不能在最大时延内完成的任务
    task_list_failure, task_fleet = clean_task(fleet=task_fleet)

    # 计算任务优先级-->优先级队列  相同则按T_max_i: 小-->大
    task_pri_dict = calcu_task_pri_in_list(task_fleet_const=task_fleet.pri_j_task_fleet_const,
                                           task_list=task_fleet.ini_list, para=para_abv)
    df = pd.DataFrame([task_pri_dict])  # df 好处理
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df2 = df2.sort_values(by=0, ascending=False)
    one_ves = Ves(volume_vec_f)

    # 完不成的不考虑
    df2 = df2.drop(index=task_list_failure)

    while df2.empty is False:  # 存在未调度的任务
        # 优先级最高任务
        num_task = df2.index[0]
        # ves是否立即满足计算需求
        task_volume = task_fleet.pri_j_task_fleet_const[num_task][1]
        # 容量是否满足
        if one_ves.rest_volume >= task_volume:
            # 任务等待时间
            task_fleet.task_wait_t[num_task] = one_ves.current - 0
            # 任务的计算时间
            task_fleet.task_calcu_t[num_task] = get_calcu_time(Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
                                                               f_max=volume_vec_f)
            # 完成时间
            task_fleet.task_finish_t[num_task] = task_fleet.task_wait_t[num_task] + task_fleet.task_calcu_t[num_task]
            task_finish_t = pd.Series(task_fleet.task_finish_t)
            # 更新ves
            # 1.任务剩余完成时间队列
            one_ves.task_rest_t_list[num_task] = task_fleet.task_calcu_t[num_task]
            # 2.更新优先级队列
            df2 = df2.drop(index=num_task)
            # 3.更新ves容量
            one_ves.rest_volume -= task_volume
        else:  # 当前容量不足
            data = pd.Series(one_ves.task_rest_t_list)
            series = data.sort_values(ascending=True)
            min_key = series.keys()[0]
            # 1.更新当前时刻
            one_ves.current += series[min_key]
            # 2.剔除一个任务
            del one_ves.task_rest_t_list[min_key]
            # 3.释放空间
            one_ves.rest_volume += task_fleet.pri_j_task_fleet_const[min_key][1]
            # 4.更新任务剩余完成时间队列
            task_rest_t_list = {}
            for key in one_ves.task_rest_t_list:
                one_ves.task_rest_t_list[key] = one_ves.task_rest_t_list[key] - series[min_key]
                rest_t = one_ves.task_rest_t_list[key]
                task_rest_t = pd.Series(one_ves.task_rest_t_list)
    # 计算失败率
    f_fate = calcu_failure_rate(fleet=task_fleet, first_failure_list=task_list_failure)
    print('任务分配完毕')

    if n_pic in ['pic1', 'pic4', 'pic3', 'pic5']:
        return f_fate  # '任务失败率'


# alg2 更新任务优先级
def update_pri(fleet, rest_task_list):
    task_pri_dict = {}
    for n_task in rest_task_list:
        task_pri_dict[n_task] = (fleet.task_wait_t_before_assign[n_task] + fleet.task_calcu_t_before_assign[n_task]) / \
                                fleet.task_calcu_t_before_assign[n_task]

    return task_pri_dict


def alg2_func(task_fleet, n_pic, n_task, volume_vec_f, t_dev, T_MAX_i, para_abv):
    """动态最高任务优先级  抢占式调度-优先级有变化-一个公式"""
    # 计算不能在最大时延内完成的任务
    task_list_failure, task_fleet = clean_task(fleet=task_fleet)
    # 计算任务的执行时间
    for num_task in task_fleet.ini_list:
        task_fleet.task_calcu_t_before_assign[num_task] = get_calcu_time(
            Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
            f_max=volume_vec_f)
    # 计算任务优先级-->优先级队列  相同则按T_max_i: 小-->大
    task_pri_dict = calcu_task_pri_in_list(task_fleet_const=task_fleet.pri_j_task_fleet_const,
                                           task_list=task_fleet.ini_list, para=para_abv)
    df = pd.DataFrame([task_pri_dict])  # df 好处理
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df2 = df2.sort_values(by=0, ascending=False)
    one_ves = Ves(volume_vec_f)

    # 完不成的不考虑
    df2 = df2.drop(index=task_list_failure)

    while df2.empty is False:  # 存在未调度的任务
        # 更新任务优先级
        rest_list = list(df2.index)
        # 更新任务等待时间
        for num in rest_list:
            task_fleet.task_wait_t_before_assign[num] = one_ves.current
        task_pri_dict = update_pri(fleet=task_fleet, rest_task_list=rest_list)
        df = pd.DataFrame([task_pri_dict])  # df 好处理
        df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        df2 = df2.sort_values(by=0, ascending=False)

        # 优先级最高任务
        num_task = df2.index[0]
        # ves是否立即满足计算需求
        task_volume = task_fleet.pri_j_task_fleet_const[num_task][1]
        # 容量是否满足
        if one_ves.rest_volume >= task_volume:
            # 任务等待时间
            task_fleet.task_wait_t[num_task] = one_ves.current - 0
            # 任务的计算时间
            task_fleet.task_calcu_t[num_task] = get_calcu_time(Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
                                                               f_max=volume_vec_f)
            # 完成时间
            task_fleet.task_finish_t[num_task] = task_fleet.task_wait_t[num_task] + task_fleet.task_calcu_t[num_task]
            task_finish_t = pd.Series(task_fleet.task_finish_t)
            # 更新ves
            # 1.任务剩余完成时间队列
            one_ves.task_rest_t_list[num_task] = task_fleet.task_calcu_t[num_task]
            # 2.更新优先级队列
            df2 = df2.drop(index=num_task)
            # 3.更新ves容量
            one_ves.rest_volume -= task_volume
        else:  # 当前容量不足
            data = pd.Series(one_ves.task_rest_t_list)
            series = data.sort_values(ascending=True)
            min_key = series.keys()[0]
            # 1.更新当前时刻
            one_ves.current += series[min_key]
            # 2.剔除一个任务
            del one_ves.task_rest_t_list[min_key]
            # 3.释放空间
            one_ves.rest_volume += task_fleet.pri_j_task_fleet_const[min_key][1]
            # 4.更新任务剩余完成时间队列
            task_rest_t_list = {}
            for key in one_ves.task_rest_t_list:
                one_ves.task_rest_t_list[key] = one_ves.task_rest_t_list[key] - series[min_key]
                rest_t = one_ves.task_rest_t_list[key]
                task_rest_t = pd.Series(one_ves.task_rest_t_list)

    # 计算失败率
    f_fate = calcu_failure_rate(fleet=task_fleet, first_failure_list=task_list_failure)

    print('任务分配完毕')

    if n_pic in ['pic1', 'pic4', 'pic3', 'pic5']:
        return f_fate  # '任务失败率'


def find_a_task(volume_task, task_fleet, one_ves):
    for task in one_ves.task_rest_t_list:
        num_task = int(task)
        task_volume = task_fleet.pri_j_task_fleet_const[num_task][1]
        if task_volume >= volume_task:
            return num_task, task_volume
    return None, None


def replace(task_fleet, one_ves, num_task, task_replaced):


    return one_ves


def alg3_func(task_fleet, n_pic, n_task, volume_vec_f, T_MAX_i, t_dev, para_abv):
    """最高任务优先级  直接抢占式调度-不考虑当前任务状态"""

    # 计算不能在最大时延内完成的任务
    task_list_failure, task_fleet = clean_task(fleet=task_fleet)
    # 计算任务的执行时间
    for num_task in task_fleet.ini_list:
        task_fleet.task_calcu_t_before_assign[num_task] = get_calcu_time(
            Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
            f_max=volume_vec_f)
    # 计算任务优先级-->优先级队列  相同则按T_max_i: 小-->大
    task_pri_dict = calcu_task_pri_in_list(task_fleet_const=task_fleet.pri_j_task_fleet_const,
                                           task_list=task_fleet.ini_list, para=para_abv)
    df = pd.DataFrame([task_pri_dict])  # df 好处理
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df2 = df2.sort_values(by=0, ascending=False)  # 按优先级排列
    one_ves = Ves(volume_vec_f)
    # 初始化等待时间
    for num in task_fleet.ini_list:
        task_fleet.task_wait_t_before_assign[num] = 0
    # 完不成的不考虑
    df2 = df2.drop(index=task_list_failure)
    times = 0  # 切换次数
    index_t_dev = 0
    while df2.empty is False:  # 存在未调度的任务

        rest_list = list(df2.index)
        # 更新任务等待时间
        for num in rest_list:
            task_fleet.task_wait_t_before_assign[num] = one_ves.current
        # 计算任务优先级
        task_pri_dict = update_pri_in_my_alg(fleet=task_fleet, rest_task_list=rest_list)
        df = pd.DataFrame([task_pri_dict])  # df 好处理
        df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        df2 = df2.sort_values(by=0, ascending=False)
        # 判断是否存在存在满足调整优先级条件的task
        df2, exist = judge_rest_task(task_fleet=task_fleet, task_pri_df=df2)
        # 优先级最高任务
        num_task = df2.index[0]
        # ves是否立即满足计算需求
        task_volume = task_fleet.pri_j_task_fleet_const[num_task][1]
        # 容量是否满足
        if one_ves.rest_volume >= task_volume:
            # 任务等待时间
            task_fleet.task_wait_t[num_task] = one_ves.current - 0
            # 任务的计算时间
            task_fleet.task_calcu_t[num_task] = get_calcu_time(Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
                                                               f_max=volume_vec_f)
            # 完成时间
            task_fleet.task_finish_t[num_task] = task_fleet.task_wait_t[num_task] + task_fleet.task_calcu_t[num_task]
            task_finish_t = pd.Series(task_fleet.task_finish_t)
            # 更新ves
            # 1.任务剩余完成时间队列
            one_ves.task_rest_t_list[num_task] = task_fleet.task_calcu_t[num_task]
            # 2.更新优先级队列
            df2 = df2.drop(index=num_task)
            # 3.更新ves容量
            one_ves.rest_volume -= task_volume
        else:  # 当前容量不足
            if exist is True and one_ves.current > t_dev[index_t_dev]:  # 尝试直接抢占
                index_t_dev += 1
                # 找到一个可以被抢占的任务，容量满足
                task_replaced, task_volume_replace = find_a_task(task_volume, task_fleet, one_ves)

                if task_replaced is not None:  # 存在可被替换
                    one_ves = replace(task_fleet, one_ves, num_task, task_replaced)  # 替换
                    times += 1

                    # --基站中替换--
                    one_ves.task_rest_t_list.pop(task_replaced)
                    # 任务的计算时间
                    task_fleet.task_calcu_t[num_task] = get_calcu_time(
                        Ci=task_fleet.pri_j_task_fleet_const[num_task][1],
                        f_max=volume_vec_f)
                    # 更新ves
                    # 1.任务剩余完成时间队列
                    one_ves.task_rest_t_list[num_task] = task_fleet.task_calcu_t[num_task]
                    # --被替换任务放入优先级list中--
                    # df2.append(0.1, index=task_replaced)
                    df2.loc[task_replaced] = 0.1
                    # 2.更新优先级队列
                    df2 = df2.drop(index=num_task)
                    # 3.更新ves容量
                    one_ves.rest_volume -= task_volume
                    one_ves.rest_volume += task_volume_replace

                else:  # 不
                    continue

                # 如果存在，则替换出
                # 更新ves
            else:
                data = pd.Series(one_ves.task_rest_t_list)
                series = data.sort_values(ascending=True)
                try:
                    min_key = series.keys()[0]
                except IndexError:
                    print('err:', IndexError)
                # 1.更新当前时刻
                one_ves.current += series[min_key]
                # 2.剔除一个任务
                del one_ves.task_rest_t_list[min_key]
                # 3.释放空间
                one_ves.rest_volume += task_fleet.pri_j_task_fleet_const[min_key][1]
                # 4.更新任务剩余完成时间队列
                task_rest_t_list = {}
                for key in one_ves.task_rest_t_list:
                    one_ves.task_rest_t_list[key] = one_ves.task_rest_t_list[key] - series[min_key]
                    rest_t = one_ves.task_rest_t_list[key]
                    task_rest_t = pd.Series(one_ves.task_rest_t_list)

    print('任务分配完毕')
    if n_pic == 'pic4':
        return times  # '任务切换次数'
