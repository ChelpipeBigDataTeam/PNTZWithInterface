import pandas as pd
import numpy as np
import pickle
import json
from app.my_libs.calc_features import *
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

ls_columns_result = [
    'Врем. сопротивление',
    'прогноз Врем. сопротивление',
    'MAE Врем. сопротивление',
    # 'MSE Врем. сопротивление',
    'Предел текучести',
    'прогноз Предел текучести',
    'MAE Предел текучести',
    # 'MSE Предел текучести',
]

ls_columns_output = [
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

load = datetime.now()
print(os.getcwd())
database = pd.read_csv('app/DATA/prepared_to_saw_gp.csv')
print('База исторических режимов загружена за ', datetime.now() - load)

cleaned_input = pd.DataFrame()


def close_value(df, row, value):
    s_close = df[row].value_counts().index[0]
    min_delta = np.abs(s_close - value)
    for s_ in df[row].value_counts().index:
        delta = np.abs(s_ - value)
        if delta < min_delta:
            min_delta = delta
            s_close = s_
    return s_close


def find_close_sort(df):
    tmp = pd.DataFrame()
    for i in range(len(df.index)):
        answ = df.iloc[i, :].copy()
        mark = answ['марка стали']
        d = answ['диаметр']
        s = answ['толщина стенки']
        gp = answ['Гр. прочн.']
        answ = database[database['Гр. прочн.'] == gp]
        if answ[answ['марка стали'] == mark].shape[0] == 0:
            mark = mark.split('-')[0]
        answ = answ[answ['марка стали'] == mark]
        if answ[answ['диаметр'] == d].shape[0] != 0:
            answ = answ[answ['диаметр'] == d]
            s_close = close_value(answ, 'толщина стенки', s)
            answ = answ[answ['толщина стенки'] == s_close]
        elif answ[answ['толщина стенки'] == s].shape[0] != 0:
            answ = answ[answ['толщина стенки'] == s]
            d_close = close_value(answ, 'диаметр', d)
            answ = answ[answ['диаметр'] == d_close]
        answ = answ[answ['Дата термообработки'] == answ['Дата термообработки'].max()][:1]
        tmp = pd.concat([tmp, answ])
        if answ.shape[0] == 0:
            with open('reason_del.txt', 'a', encoding='utf-8') as f:
                f.write('Базовый режим для строки '+ str(i) + ' не найден.')
            f.close()
            # print('Базовый режим для строки ', i, ' не найден.')
            missing = ['-']
            missing = pd.DataFrame(missing)
            tmp = pd.concat([tmp, missing])
    return tmp


# возвращает модель, список признаков, по которому строилась модель и скейлер
def load_model(dir_name, target):
    model = pickle.load(open(dir_name + '/RF_model_' + target + '.sav', 'rb'))
    ls_need_col = json.load(open(dir_name + '/ls_need_col', "r"))
    scaler = StandardScaler()
    scale_data = json.load(open(dir_name + '/scaler', "r"))
    scaler.mean_ = scale_data[0]
    scaler.scale_ = scale_data[1]
    return model, ls_need_col, scaler


def make_result_valid_file(file_name, dir_names, targets, output_filename):
    """Сохраняет файл с результатами """
    result = pd.DataFrame()
    base = pd.DataFrame()
    k = 0
    for target, dir_name in zip(targets, dir_names):
        model, ls_need_col, scaler = load_model(dir_name, target)
        time = datetime.now()
        if k == 0:
            try:
                valid = clean_data(file_name, ls_columns_output)
                valid.reset_index(drop=True, inplace=True)
            except KeyError:
                valid = clean_data(file_name, ls_columns_output + ['№ партии', "марка стали"])
                valid.reset_index(drop=True, inplace=True)
            cleaned_input = valid.copy()
            print('Введенные данные очищены за ', datetime.now() - time)
        else:
            valid = cleaned_input.copy()
        time = datetime.now()
        try:
            y_valid = valid[target].copy()
        except KeyError:
            y_valid = [0 for x in range(0, valid.shape[0])]
        try:
            X_valid = valid[ls_columns_output].copy()
        except KeyError:
            X_valid = valid[ls_need_col].copy()
        X_valid_c = X_valid[ls_need_col].copy()
        X_valid_c = scaler.transform(X_valid_c)
        time = datetime.now()
        y_pred = model.predict(X_valid_c)
        print('Предсказания ', y_pred.shape[0], ' строчек за ', datetime.now() - time)
        result[target] = y_valid
        result['прогноз ' + target] = y_pred
        for x in range(result[target].shape[0]):
            if result.loc[x, target] != 0:
                result.loc[x, 'MAE ' + target] = np.abs(result.loc[x, target] - result.loc[x, 'прогноз ' + target])
            else:
                result.loc[x, 'MAE ' + target] = ''
        k += 1

    time = datetime.now()
    base = find_close_sort(X_valid)
    print('Найдены базовые исторические режимы за ', datetime.now() - time)
    time = datetime.now()
    result = pd.concat([result, X_valid], axis=1)
    time = datetime.now()
    base = base[list(set(base.columns) & set(result.columns))]
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

def main(name):
    file_name = os.getcwd()+'/app/input/' + name
    targets = ['Предел текучести', 'Врем. сопротивление']
    dir_names = [os.getcwd() + '/app/DATA/MODELS_RF/YS 14june', os.getcwd() + '/app/DATA/MODELS_RF/H 14june']

    now = datetime.now()
    time = "%d_%ddate %d_%d_%dtime" % (now.day, now.month, now.hour, now.minute, now.second)
    output_filename = os.getcwd() + '/app/output/' + "_" + time + ".xlsx"

    time = datetime.now()
    df, y_valid, y_pred = make_result_valid_file(file_name, dir_names, targets, output_filename)
    print('Время работы скрипта на ', int(df.shape[0] / 3), 'строчках: ', datetime.now() - time)
