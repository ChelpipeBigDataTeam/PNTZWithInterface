from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import json
import scipy
from scipy.optimize import minimize, fmin_powell, fmin, differential_evolution
import warnings
from app.my_libs.calc_features import *
from sklearn.preprocessing import StandardScaler
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 100
from scipy.optimize import minimize
import os
import math
import warnings
warnings.filterwarnings("ignore")

ls_opt_need = [
    'Гр. прочн.',
    'марка стали',
    'толщина стенки',
    'диаметр'
]

targets = ['Предел текучести',
           'Врем. сопротивление']

models_bonds = [
    'Текучесть середина',
    'Прочность середина'
]

ls_columns_output = [
    '№ партии',
    '№ плавки',
    'марка стали',
    'диаметр',
    'толщина стенки',
    'Гр. прочн.',
    '1 зона по ВТР закалка',
    '2 зона по ВТР закалка',
    '3 зона по ВТР закалка',
    'шаг балок закалочная печь, сек',
    'Скорость прохождения трубы через спрейер, м/с',
    't˚ C трубы после спреера',
    '1 зона ВТР и уставка отпуск',
    '2 зона ВТР и уставка отпуск',
    '3 зона ВТР и уставка отпуск',
    '4 зона ВТР и уставка отпуск',
    '5 зона ВТР и уставка отпуск',
    'шаг балок отпускная печь, сек',
    'C',
    'Mn',
    'Si',
    'P',
    'S',
    'Cr',
    'Ni',
    'Cu',
    'Al',
    'V',
    'Ti',
    'Nb',
    'Mo',
    'N',
    'B',
    'C-coef',
    'Параметр закалка',
    'Параметр отпуск',
    'Параметр отпуск новый V',
    'Величина зерна',
    'Тип предела текучести (1186)',
    'Дата термообработки',
    'ICD',
    'Примечание',
    'Параметр отпуск новый',
    'Параметр отпуск новый 2'
#     'длина трубы'
    ]

ls_opt_need = [
    'Гр. прочн.',
    'марка стали',
    'толщина стенки',
    'диаметр'
]

bounds = [
    'Предел текучести нижняя граница',
    'Предел текучести верхняя граница',
    'Предел прочности нижняя граница',
    'Предел прочности верхняя граница'
]

ls_additional = [
    '1 зона по ВТР закалка',
    '2 зона по ВТР закалка',
    '3 зона по ВТР закалка',
    'Скорость прохождения трубы через спрейер, м/с',
    '1 зона ВТР и уставка отпуск',
    '2 зона ВТР и уставка отпуск',
    '3 зона ВТР и уставка отпуск',
    '4 зона ВТР и уставка отпуск',
    '5 зона ВТР и уставка отпуск',
    'шаг балок закалочная печь, сек',
    'шаг балок отпускная печь, сек',
    'Тип предела текучести (1186)'
]


ls_temp_z = [
    '1 зона по ВТР закалка', '2 зона по ВТР закалка',
    '3 зона по ВТР закалка'
]

ls_temp_o = [
    '1 зона ВТР и уставка отпуск',
    '2 зона ВТР и уставка отпуск', '3 зона ВТР и уставка отпуск',
    '4 зона ВТР и уставка отпуск', '5 зона ВТР и уставка отпуск'
]

# ls_s_spr = ['Скорость прохождения трубы через спрейер, м/с']

ls_shag = [
    'шаг балок закалочная печь, сек', 'шаг балок отпускная печь, сек'
]

ls_fit_param = ls_temp_o+ls_temp_z+ls_shag

replace_dict_gr = {
    ' ':'',
    '/':'-',
    'ТИП':'тип',
    'К':'K', #русский на английский, везде
    'С':'C',
    'Р':'P',
    'Х':'X',
    'Е':'E',
    'Т':'T',
    'М':'M',
    'У':'Y',
    'Н':'H',
    'В':'B',
    'А':'A',
    'П':'n',
    'О':'O',
    'Т':'T'
}


def load_model(dir_name):
    model = pickle.load(open(dir_name + '/RF_model_.sav', 'rb'))
    ls_need_col = json.load(open(dir_name + '/ls_need_col', "r"))
    try:
        scaler = StandardScaler()
        scale_data = json.load(open(dir_name + '/scaler', "r"))
        scaler.mean_ = scale_data[0]
        scaler.scale_ = scale_data[1]
    except:
        scaler = None
    return model, ls_need_col, scaler


def fill_up(x):
    if pd.isnull(x['Предел текучести верхняя граница']):
        x['Предел текучести верхняя граница'] = x['Предел текучести нижняя граница'] + 30
    if pd.isnull(x['Предел прочности верхняя граница']):
        x['Предел прочности верхняя граница'] = x['Предел прочности нижняя граница'] + 30
    return x


def fix_h_group(st):
    st = str(st)
    st = st.upper()
    for it in replace_dict_gr:
        st = st.replace(it, replace_dict_gr[it])
    return st


def close_value(database, col, value):
    database['diff'] = np.abs(database[col] - value)
    return database.loc[(database['diff']).argmin(), :][col]


def find_row_close_sort(database, row, ls_need_col):
    for col in ls_opt_need:
        tmp = database[database[col] == row[col]]
        if tmp.shape[0] > 0:
            database = tmp
        else:
            try:
                value = close_value(database, col, row[col])
                database = database[database[col] == value]
            except TypeError:
                database[col + '_'] = database[col].apply(lambda x: x.split('-')[0])
                #     return database
                if col == 'Гр. прочн.':
                    row['Гр. прочн.'] = fix_h_group(row['Гр. прочн.'])
                tmp_copy = database[database[col] == row[col].split('-')[0]].copy()

                if tmp_copy.shape[0] == 0:
                    #                     print('++++++++++++++++++++')
                    #                     print(database)
                    #                     print('++++++++++++++++++++')
                    #                     print(row[col].split('-')[0])
                    #                     print('++++++++++++++++++++')
                    tmp_copy = database[database[col + '_'] == row[col].split('-')[0]].copy()
                database = tmp_copy.copy()

                if database.shape[0] == 0:
                    tmp = row
                    tmp[[col for col in row.index if col not in ls_opt_need]] = None
                    tmp['№ партии'] = 'Error!!! (причина:' + col + ')'

                    return tmp

    database = database.dropna(subset=ls_need_col)
    row_new = pd.Series(database[database['Дата термообработки'] == database['Дата термообработки'].max()].iloc[0, :])
    row_new[ls_opt_need] = row[ls_opt_need].copy()
    return row_new


def find_close_sort(database, df, ls_need_col):
    df['Гр. прочн.'] = df['Гр. прочн.'].apply(fix_h_group)
    df = df.apply(lambda x: find_row_close_sort(database, x, ls_need_col), axis=1)
    return df


def get_index_diff(df1, df2):
    return list(set(df1.index).difference(set(df2.index)))


def is_in_bounds(row, bounds):
    penalty = 0
    for i in range(len(bounds)):
        if bounds[i][0] <= row.iloc[:, i].values <= bounds[i][1]:
            pass
        else:
            penalty += 100
    return penalty


def model_pr(fit_params, all_params, bounds, eta, model, ls_need_col):
    centr_ys = all_params['Текучесть середина']
    centr_h = all_params['Прочность середина']
    all_params = pd.concat([all_params[list(set(ls_need_col + ['длина трубы']) - set(ls_fit_param))],
                            pd.Series(fit_params, index=ls_fit_param)])
    fit_params = pd.DataFrame(fit_params, index=ls_fit_param).T
    score = 0
    # пересчет параметров
    if pd.DataFrame(all_params).shape[0] > pd.DataFrame(all_params).shape[1]:
        all_params = pd.DataFrame(all_params).T
    else:
        all_params = pd.DataFrame(all_params)
    all_params = calc_all_features(all_params)
    all_params = new_spr(all_params)
    # попытки победить ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
    all_params.reset_index(inplace=True, drop=True)
    all_params.dropna(inplace=True)
    all_params = all_params.astype(np.float32)

    pred = model.predict(all_params[ls_need_col])
    tmp_score = np.abs(pred[:, 0] - centr_ys) + np.abs(pred[:, 1] - centr_h)
    if tmp_score < 2:
        tmp_score = 0
    if tmp_score > 5:
        #             tmp_score+=float('inf')
        tmp_score += 100
    score += tmp_score

    score += max(np.abs(all_params['шаг балок закалочная печь, сек'].values - 24),
                 np.abs(all_params['шаг балок отпускная печь, сек'].values - 24)) * eta
    score += is_in_bounds(fit_params, bounds)
    return score


def common(
        file,  # имя входного файла
        model_dir_name,  # путь к директории с моделью + ее имя
        saw_db_filename,  # путь к файлу для поиска исторических режимов + его название
        input_part,  # путь к папке куда будут сохраняться запросы
        output_part,  # путь к папке куда будут сохраняться ответы оптимизатора
        username
):
    table_for_optimize = pd.read_excel(file, skiprows=1)
    print(table_for_optimize.shape)
    if table_for_optimize.shape[0] == 0:
        return 165, True
    now = datetime.now()
    time = "%d_%ddate %d_%d_%dtime" % (now.day, now.month, now.hour, now.minute, now.second)
    input_filename = os.getcwd() + input_part + "optimizer_input_" + time + "_"+ username + ".xlsx"
    table_for_optimize.to_excel(input_filename)

    model, ls_need_col, scaler = load_model(model_dir_name)
    table_for_optimize = table_for_optimize.apply(fill_up, axis=1)

    table_for_optimize['Прочность середина'] = (table_for_optimize[
                                                    'Предел прочности нижняя граница'] + table_for_optimize[
                                                    'Предел прочности верхняя граница']) / 2.0
    table_for_optimize['Текучесть середина'] = (table_for_optimize[
                                                    'Предел текучести нижняя граница'] + table_for_optimize[
                                                    'Предел текучести верхняя граница']) / 2.0
    print(table_for_optimize.shape)
    if table_for_optimize.shape[0] == 0:
        return 165, True
    database = pd.read_csv(saw_db_filename, index_col=0)
    database = calc_all_features(database)
    answ = find_close_sort(database, table_for_optimize, ls_need_col)
    if (answ['№ партии'].dtype == object or answ['№ партии'].dtype == str):
        err1 = answ[answ['№ партии'] == 'Error!!! (причина:Гр. прочн.)']
        err2 = answ[answ['№ партии'] == 'Error!!! (причина:марка стали)']
    else:
        err1 = []
        err2 = []
    errors = []
    if len(err1) != 0:
        for i in range(len(err1)):
            errors.append(err1.index[i])
    if len(err2) != 0:
        for i in range(len(err2)):
            errors.append(err2.index[i])
    for i in range (len(errors)):
        answ = answ[~(answ.index == errors[i])]
        table_for_optimize = table_for_optimize[~(table_for_optimize.index == errors[i])]
    answ = answ.reset_index()
    table_for_optimize = table_for_optimize.reset_index()
    print(answ)
    print(table_for_optimize)

    if answ.shape[0] == 0:
        return errors, True
    try_ = table_for_optimize.copy()
    try_ = try_.dropna()

    diff = get_index_diff(answ, try_)

    answ = pd.concat([answ.take(diff), try_])
    #     print(answ.columns)
    """""""""""""""""""""""""""
    """""""""""""""""""""""""""
    try:
        answ = answ[ls_columns_output]
    except KeyError:
        return errors, True

    answ = pd.concat([answ, table_for_optimize[['Прочность середина', 'Текучесть середина']]], axis=1)

    answ = mean_chem(answ)

    answ = answ[~answ['C'].isnull()]
    answ[ls_chem] = answ[ls_chem].fillna(0)

    answ = calc_all_features(answ)
    answ = len_pipe(answ)
    answ = calc_AC(answ)

    answ.reset_index(inplace=True, drop=True)
    print(answ.shape)
    if answ.shape[0] == 0:
        return errors, True
    # Температура трубы после спреера!!!!!!!! Нужна для модели, чатсо ее нет, заполняю как 70, чтобы это хоть както работало
    # Исправить когда будет время!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """TODO"""
    answ['t˚ C трубы после спреера'] = answ['t˚ C трубы после спреера'].fillna(70)
    X_c_down = calc_all_features(answ[ls_need_col + ['AC3', 'AC1']]).dropna().copy()
    print(X_c_down.shape)
    for col in ls_temp_z:
        X_c_down[col] = X_c_down.apply(lambda x: max(x['AC3'], x[col] - 50), axis=1)

    for el in ls_temp_o:
        X_c_down[el] = X_c_down.apply(lambda x: max(450, x[el] - 50), axis=1)

    for el in ls_shag:
        X_c_down[el] = X_c_down.apply(lambda x: max(24, x[el] - 8), axis=1)

    # X_c_down['Скорость прохождения трубы через спрейер, м/с'] = X_c_down[
    #     'Скорость прохождения трубы через спрейер, м/с'].apply(lambda x: x if x>0 else 0)

    X_c_up = calc_all_features(answ[ls_need_col + ['AC3', 'AC1']]).dropna().copy()

    for el in ls_temp_z:
        X_c_up[el] = X_c_up.apply(lambda x: min(990, x[el] + 50), axis=1)

    for col in ls_temp_o:
        X_c_up[col] = X_c_up.apply(lambda x: min(x['AC1'], x[col] - 50), axis=1)

    answ_n = answ[ls_need_col + models_bonds + ['длина трубы']].dropna()

    answ_n = answ[ls_columns_output + models_bonds + ['длина трубы']]

    from scipy.optimize import brute, basinhopping

    ls_answ_n = ls_columns_output + models_bonds + ['длина трубы']
    ls_answ_n.remove('Примечание')
    answ_n = answ[ls_answ_n]
    answ_n = answ_n
    # .dropna()
    answ_n

    answers_array = []
    # tmp_res = []
    X_a = calc_all_features(X_c_down[ls_need_col]).copy()
    X_b = calc_all_features(X_c_up[ls_need_col]).copy()
    answ_n.reset_index(inplace=True, drop=True)
    X_a.reset_index(inplace=True, drop=True)
    X_b.reset_index(inplace=True, drop=True)
    answ_n.reset_index(inplace=True, drop=True)
    for it in X_a.index:
        bounds = [(i, j) for i, j in zip(X_a.loc[it, ls_fit_param], X_b.loc[it, ls_fit_param])]
        print(answ_n.iloc[it, -2], answ_n.iloc[it, -3])
        all_params = answ_n.iloc[it, :]
        fit_p = answ_n.loc[it, ls_fit_param]

        #     a = minimize(lambda fit_params: model_pr(fit_params,
        #                                              all_params,
        #                                              bounds,
        #                                              10),
        #                  fit_p, method='TNC', bounds=bounds)

        a = fmin(lambda fit_params: model_pr(fit_params,
                                             all_params,
                                             bounds,
                                             10,
                                             model,
                                             ls_need_col),
                 fit_p)

        #     a = differential_evolution(lambda fit_params: model_pr(fit_params,all_params,bounds,10), bounds)

        #     a = basinhopping(lambda fit_params: model_pr(fit_params,all_params,bounds,10), fit_p)
        quA = model.predict(X_a.loc[it, ls_need_col].values.reshape(1, -1))
        #     return answ_n
        quO = model.predict(answ_n.loc[it, ls_need_col].values.reshape(1, -1))

        quB = model.predict(X_b.loc[it, ls_need_col].values.reshape(1, -1))
        print('ВАПРОС a', it, quA[:, 0], quA[:, 1])
        print('ВАПРОС origin', it, quO[:, 0], quO[:, 1])
        print('ВАПРОС b', it, quB[:, 0], quB[:, 1])
        print(a)
        #     tmp_res.append(all_params)
        all_params = pd.concat([all_params[set(
            ls_columns_output + ['длина трубы', 'Текучесть середина', 'Прочность середина']) - set(ls_fit_param)],
                                pd.Series(a, index=ls_fit_param)])

        answers_array.append(all_params)
        #     tmp_a_x = a['x']
        #     df_h = np.concatenate((tmp_a_x[:2],tmp_a_x[3:-2],tmp_a_x[-1:]))
        #     df_ys = tmp_a_x[:-1]
        pred = model.predict(calc_all_features(pd.DataFrame(answers_array[-1]).T)[ls_need_col])

        answers_array[-1] = pd.concat(
            [answers_array[-1], pd.Series([pred[0, 0], pred[0, 1]], index=['pred Прочность', 'pred Текучесть'])])
        print('АТВЕТ ', it, pred)

    opt_answer_end = pd.DataFrame(answers_array)
    opt_answer_end = calc_all_features(opt_answer_end)
    opt_answer_end = new_spr(opt_answer_end)
    # opt_answer_end = opt_answer_end[ls_need_cols[0]+['pred Текучесть','pred Прочность']].copy()
    output_df = opt_answer_end[ls_answ_n + ['pred Текучесть', 'pred Прочность']].copy()

    output_filename = os.getcwd() + output_part + "optimizer_output_" + time + "_" + username + ".xlsx"
    output_df.to_excel(output_filename)
    return errors, False

def main(file, username):
    return common(file,
           'app/DATA/MODELS_RF/H+YS+BATH GS',
           'app/DATA/prepared_to_saw_gp_del_bath.csv',
           '/app/input/',
           '/app/output/',
            username)
