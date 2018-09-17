ls_columns_result = [
    'Врем. сопротивление',
    'прогноз Врем. сопротивление',
    'MAE Врем. сопротивление',
    #     'MSE Врем. сопротивление',
    'Предел текучести',
    'прогноз Предел текучести',
    'MAE Предел текучести',
    #     'MSE Предел текучести',
]

ls_columns_output = [
    # 'Дата термообработки',
    '№ партии',
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
    'Параметр отпуск новый',
    'Параметр отпуск новый 2',
    'Параметр отпуск новый V',
    'Величина зерна',
    'Тип предела текучести (1186)',
    'ICD'
]

ls_columns_required = [
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
    'Тип предела текучести (1186)'
]

ls_opt_need = ['Гр. прочн.', 'марка стали', 'толщина стенки', 'диаметр']

import pandas as pd
import numpy as np
import pickle
import json
from app.my_libs.calc_features import *
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

load = datetime.now()
database = pd.read_csv('app/DATA/prepared_to_saw_gp.csv', low_memory=False)
print('База исторических режимов загружена за ', datetime.now() - load)

replace_dict_gr = {
    ' ': '',
    '/': '-',
    'ТИП': 'тип',
    'К': 'K',  # русский на английский, везде
    'С': 'C',
    'Р': 'P',
    'Х': 'X',
    'Е': 'E',
    'Т': 'T',
    'М': 'M'
}


def fix_h_group(st):
    st = str(st)
    st = st.upper()
    for it in replace_dict_gr:
        st = st.replace(it, replace_dict_gr[it])
    return st


def close_value(database, col, value):
    database['diff'] = np.abs(database[col] - value)
    return database.loc[(database['diff']).argmin(), :][col]


def find_row_close_sort(database, row):
    for col in ls_opt_need:
        tmp = database[database[col] == row[col]]
        if tmp.shape[0] > 0:
            database = tmp.copy()
        else:
            try:
                value = close_value(database, col, row[col])
                database = database[database[col] == value]
            except TypeError:
                database = database[database[col].apply(lambda x: str(x).split('-')[0]) == row[col].split('-')[0]]
                #                 database = database[database[col].apply(lambda x: str(x).split(' ')[0]) == row[col].split(' ')[0]]
                if database.shape[0] == 0:
                    tmp = row
                    tmp[[col for col in row.index if col not in ls_opt_need]] = None
                    tmp['№ партии'] = 'Error!!! (причина:' + col + ')'
                    return tmp

    return pd.Series(database[database['Дата термообработки'] == database['Дата термообработки'].max()].iloc[0, :])


def find_close_sort(database, df):
    database['Гр. прочн.'] = database['Гр. прочн.'].apply(fix_h_group)
    df['Гр. прочн.'] = df['Гр. прочн.'].apply(fix_h_group)
    df = df.apply(lambda x: find_row_close_sort(database, x), axis=1)
    return df


# возвращает модель, список признаков, по которому строилась модель
def load_model(dir_name):
    model = pickle.load(open(dir_name + '/RF_model_' + '.sav', 'rb'))
    ls_need_col = json.load(open(dir_name + '/ls_need_col', "r"))
    return model, ls_need_col


<<<<<<< HEAD
def make_result_valid_file(file_name, dir_names, targets, output_filename, username):
=======
def make_result_valid_file(file_name, dir_names, targets, output_filename):
>>>>>>> f308dd357f9e616f12cb69b79c15d41170bc56b4
    """Сохраняет файл с результатами """
    result = pd.DataFrame()
    base = pd.DataFrame()
    model = []
    ls_need_col = []

    model, ls_need_col = load_model(dir_names)
    time = datetime.now()
<<<<<<< HEAD
    valid = clean_data(file_name, ls_columns_output, username)
=======
    valid = clean_data(file_name, ls_columns_output)
>>>>>>> f308dd357f9e616f12cb69b79c15d41170bc56b4
    valid.reset_index(drop=True, inplace=True)

    print('Введенные данные очищены за ', datetime.now() - time)
    time = datetime.now()

    try:
        y_valid = valid[targets].copy()
    except KeyError:
        y_valid = [0 for x in range(0, valid.shape[0])]
    try:
        X_valid = valid[ls_columns_output].copy()
    except KeyError:
        X_valid = valid[ls_need_col].copy()

    X_valid_c = X_valid[ls_need_col].copy()
    time = datetime.now()

    y_pred = model.predict(X_valid_c)
    print('Предсказания ', y_pred.shape[0], ' строчек за ', datetime.now() - time)

    for i in range(len(targets)):
        result[targets[i]] = y_valid[targets[i]]
        result['прогноз ' + targets[i]] = y_pred[:, i]
        for x in range(result[targets[i]].shape[0]):
            if result.loc[x, targets[i]] != 0:
                result.loc[x, 'MAE ' + targets[i]] = np.abs(
                    result.loc[x, targets[i]] - result.loc[x, 'прогноз ' + targets[i]])
            else:
                result.loc[x, 'MAE ' + targets[i]] = ''

    time = datetime.now()
    base = find_close_sort(database, X_valid)
    print('Найдены базовые исторические режимы за ', datetime.now() - time)
    time = datetime.now()
    result = pd.concat([result, X_valid], axis=1)
    time = datetime.now()
    base.index = [str(i) + ' исторический' for i in range(len(base.index))]
    con = pd.DataFrame()
    time = datetime.now()
    missing = ['-' for i in result.index]
    missing = pd.DataFrame(missing)
    for i, j, k in zip(result.index, base.index, missing.index):
        con = pd.concat([con, result[result.index == i], base[base.index == j], missing[missing.index == k]])
    con = con[ls_columns_result + ls_columns_output]
    con.to_excel(output_filename)
    return con, y_valid, y_pred

<<<<<<< HEAD
def main(file, username):
    file_name = file
    targets = ['Врем. сопротивление', 'Предел текучести']
=======
def main(file):
    file_name = file
    targets = ['Предел текучести', 'Врем. сопротивление']
>>>>>>> f308dd357f9e616f12cb69b79c15d41170bc56b4
    dir_names = os.getcwd() + '/app/DATA/MODELS_RF/H+YS+BATH GS'

    now = datetime.now()
    time = "%d_%ddate %d_%d_%dtime" % (now.day, now.month, now.hour, now.minute, now.second)
<<<<<<< HEAD
    output_filename = os.getcwd() + '/app/output/' + "prediction_output_" + time + "_" + username + ".xlsx"

    time = datetime.now()
    df, y_valid, y_pred = make_result_valid_file(file_name, dir_names, targets, output_filename, username)
=======
    output_filename = os.getcwd() + '/app/output/' + "_" + time + ".xlsx"

    time = datetime.now()
    df, y_valid, y_pred = make_result_valid_file(file_name, dir_names, targets, output_filename)
>>>>>>> f308dd357f9e616f12cb69b79c15d41170bc56b4
    print('Время работы скрипта на ', int(df.shape[0] / 3), 'строчках: ', datetime.now() - time)