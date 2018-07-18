import pandas as pd
import numpy as np

def reason_del(df):
    dict_err = {}
    for i in range(len(df.index)):
        arr_col = []
        for j, col in enumerate(df.columns):
            if pd.isnull(df.iloc[i, j]):
                arr_col.append(col)
        dict_err[i] = arr_col

    open('app/reason_del.txt', 'w', encoding='utf-8')
    with open('app/reason_del.txt', 'a', encoding='utf-8') as f:
        for i, j in zip(dict_err.keys(), dict_err.values()):
            arr_col = []
            for k in ls_columns_required:
                if k in j:
                    arr_col.append(k)
            if arr_col:
                f.write('Строка ' + str(i+1) + ' удалена из-за пустых столбцов: ' + str(arr_col) + '\n')
                
                
def mean_chem(df):
    mean = pd.read_csv('app/DATA/mean_chem_steel.csv')
    ls_mark = list(mean['марка стали'])
    for mark in ls_mark:
        if df[df['марка стали']==mark].shape[0]>0:
            tmp2 = mean.loc[mean['марка стали']==mark, ls_chem].to_dict()
            d = {i:list(tmp2[i].values())[0] for i in tmp2}
            tmp = df.loc[df['марка стали']==mark, ls_chem].fillna(d)
            df.loc[df['марка стали']==mark, ls_chem] = tmp     
    return df
    
def new_spr(df):
    for x in df['диаметр']:
        if pd.notna(x):
            if x < 114:
                df['длина трубы'][x] = df['Скорость прохождения трубы через спрейер, м/с'][0]*(df['шаг балок закалочная печь, сек'][0] - 8)
                df['min скорость спрейера'] = df['длина трубы'][0]/(df['шаг балок закалочная печь, сек']-8)
            else:
                df['длина трубы'] = df['Скорость прохождения трубы через спрейер, м/с'][0]*(2*df['шаг балок закалочная печь, сек'][0] - 8)
                df['min скорость спрейера'] = df['длина трубы'][0]/(2*df['шаг балок закалочная печь, сек']-8)
            df['Скорость прохождения трубы через спрейер, м/с'][1:] = df['min скорость спрейера'][1:].copy()
    return df

def del_bath(df):
    '''Очистка данных с закалкой в ванне'''
#     df = df[pd.isnull(df['t˚ C трубы после ванны'])]
#     df = df[pd.isnull(df['время выдержки в закалочной ванне, сек.'])]
    del df['t˚ C трубы после ванны']
    del df['время выдержки в закалочной ванне, сек.']
    return df

def clean_data(file_name, ls_need_col):
    '''Доочистка prepared файлов'''
    df = pd.read_excel(file_name)
    df['№ партии'] = df['№ партии'].apply(lambda x: str(x).replace('.0', ''))
    df = mean_chem(df)
    df = calc_all_features(df)
    try:
        df = df[ls_need_col+['Предел текучести', 'Врем. сопротивление']]
    except KeyError:
        df = df[ls_need_col]
        print('Warning!, Предел текучести, Врем. сопротивление, № партии Not in index')
    try:
        df = del_bath(df)
    except:
        pass
    print(df.shape)
    df[['Предел текучести', 'Врем. сопротивление']] = df[['Предел текучести', 'Врем. сопротивление']].fillna(0)
    reason_del(df)
    df = df.dropna()
    print(df.shape)
    return df

def clean_data_sheets(df, ls_need_col):
    '''Доочистка prepared файлов'''
    df = calc_all_features(df)
    df = df[ls_need_col+['Предел текучести', 'Врем. сопротивление', '№ партии']]
    try:
        df = del_bath(df)
    except:
        pass
    print(df.shape)
    df = df.dropna()
    print(df.shape)
    return df

def ideal_critical_d(df):
    """Считает идеальный критический диаметр, возвращает датафрейм с добавленным признаком"""
    df['k1'] = df['C'].apply(lambda x:  0.54*x if x <= 0.39 else 0.171+0.001*x+0.265*x*x)
    df['k2'] = df['Mn'].apply(lambda x: 3.3333*x+1 if x <= 1.2 else 5.1*x-1.12)
    df['ICD'] = df['k1']*df['k2']*(1+0.7*df['Si'])*(1+0.363*df['Ni'])*(1+2.16*df['Cr'])*(
        1+3.0*df['Mo'])*(1+0.365*df['Cu'])*(1+1.73*df['V'])
    del df['k1']
    del df['k2']
    return df

def calc_C_coef(df):
    """Считает углеродный коэффициент, возвращает датафрейм с добавленным признаком"""
    df['C-coef'] = (df['C'] + df['Mn']/6.0 + (df['Cr'] + df['Mo'] + df['V'])/5.0 +
                        (df['Ni'] + df['Cu'])/15.0)
    return df

def calc_quenching_param(df):
    """Считает параметр закалки, возвращает датафрейм с добавленным признаком"""
    df['Параметр закалка'] = 1/(1/(df[['3 зона по ВТР закалка']].mean(axis=1) 
                            + 273) - (2.303*1.986*np.log10(61*df['шаг балок закалочная печь, сек']/3600)/110000))-273
    return df

def calc_tempering_param(df):
    """Считает параметр отпуска, возвращает датафрейм с добавленным признаком"""
    df['Параметр отпуск'] = (df[['3 зона ВТР и уставка отпуск','4 зона ВТР и уставка отпуск','5 зона ВТР и уставка отпуск']].mean(axis=1)
                        + 273)*(20+np.log(94*df['шаг балок отпускная печь, сек']/3600))*1e-3
    return df

def calc_tempering_param_new(df):
    """Считает параметр отпуска усовершенствованный (с зависимостью от химии), возвращает датафрейм с добавленным признаком"""
    df['Параметр отпуск новый'] = (np.log(94*df['шаг балок отпускная печь, сек']/3600)-(
        (114.7*(df['Mo']+df['Mn']/5+df['Cr']/10)+46.6)/4.6/(
        df[['2 зона ВТР и уставка отпуск', '3 зона ВТР и уставка отпуск','4 зона ВТР и уставка отпуск','5 зона ВТР и уставка отпуск']].mean(axis=1)
                        + 273))+50)
    return df

def calc_tempering_param_new_2(df):
    """Считает параметр отпуска, возвращает датафрейм с добавленным признаком"""
    df['Параметр отпуск новый 2'] = (df[['3 зона ВТР и уставка отпуск','4 зона ВТР и уставка отпуск','5 зона ВТР и уставка отпуск']].mean(axis=1)
                        + 273)*((
        17.369-6.661*df['C']-1.604*df['Mn']-3.412*df['Si']-0.248*df['Ni']-1.112*df['Cr']-4.355*df['Mo'])+
        np.log(150*df['шаг балок отпускная печь, сек']/3600))*1e-3
    return df

def calc_ratio(df):
    '''считаем отношение предела текучести к пределу прочности'''
    df['Отношение'] = df['Предел текучести']/df['Врем. сопротивление']
    return df

"""Меняли длительность отпуска было 94"""


def calc_tempering_param_new_V(df):
    """Считает параметр отпуска, возвращает датафрейм с добавленным признаком"""
    df['Параметр отпуск новый V'] = (df[['3 зона ВТР и уставка отпуск','4 зона ВТР и уставка отпуск','5 зона ВТР и уставка отпуск']].mean(axis=1)
                        + 273)*((
        17.369-6.661*df['C']-1.604*df['Mn']-3.412*df['Si']-0.248*df['Ni']-1.112*df['Cr']-4.355*df['Mo']-2.6*df['V'])+
        np.log(94*df['шаг балок отпускная печь, сек']/3600))*1e-3
    return df

def calc_seed_size(df):
    chem_dict = {
        'X' : ['Ni', 'Cr', 'Mo', 'Cu', 'Al', 'V', 'Ti', 'Nb', 'N'],
        'p' : [-1.729, 0, 0.223, -7.449, 18.996, 1.35, -22.399, 148.515, -410.616],
        'q' : [-16185, -1132, 2540, -41949, 222088, 39721, -208274, 1721974, -4321929]
    }


    #   случайно поставила
    T = df[['2 зона по ВТР закалка', '3 зона по ВТР закалка']].mean(axis=1) 
    df['T'] = T
    #   случайно поставила
    t = 65

    df['sum_piXi'] = 0
    df['sum_qiXi'] = 0
    for i in range(0,len(chem_dict['X'])):
        df['sum_piXi']+=df[chem_dict['X'][i]]*chem_dict['p'][i]
        df['sum_qiXi']+=df[chem_dict['X'][i]]*chem_dict['q'][i]

    df['Величина зерна'] = np.exp(10.313+0.93*df['C']*(1-0.469*df['Cr'])+df['sum_piXi'])*(
    np.exp(-(77978+3116*df['C']+df['sum_qiXi'])/8.314/(T+273))*((t*df['шаг балок закалочная печь, сек'])**0.107))
    
    return df

def calc_all_features(df):
    """Считает и добавляет все необходимые признаки в датафрейм"""
    df = ideal_critical_d(df)
    df = calc_C_coef(df)
    df = calc_quenching_param(df)
    df = calc_tempering_param(df)
    df = calc_tempering_param_new(df)
    df = calc_tempering_param_new_2(df)
    df = calc_tempering_param_new_V(df)
#     df = calc_ratio(df)
    df = calc_seed_size(df)
#     df = calc_critical_T_after_spryer(df)
#     df = new_spr(df)
    return df

def make_prepared(df, exp_df, chem):
    """Подготавливает датафрейм"""
#     обединяем испытания с режимами и химией
    df_merge = pd.merge(exp_df, df, how ='left', on = ['№ плавки','№ партии'])
    print('объединяем испытания с режимами и химией', df_merge.shape)
#     заполняем пустую химию средней (или нулями)
    df_merge = mean_chem(df_merge)
    print('заполняем пустую химию средней (или нулями)', df_merge.shape)
#     удаляем строки без режимов
    df_merge = df_merge[~pd.isnull(df_merge['C'])]
    print('удаляем строки без химии', df_merge.shape)
#     добавляем рассчитаные признаки
    df_merge = calc_all_features(df_merge)
    print('cчитаем все признаки', df_merge.shape)
    df_merge = df_merge[df_merge['Cr'] < 10]
    print('Сr>10 del ', df_merge.shape)
    df_merge.reset_index(drop=True, inplace=True)
    print(df_merge.shape)
    return df_merge

ls_chem = [
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
    'B'
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

chem = [
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
    'W'
]

def calc_AC3_1(df):
    df[chem] = df[chem].fillna(0)
#     print(df[chem].describe())
    df['AC3_1'] = (911 - df.C*370-df.Mn*27.4+27.3*df.Si-6.35*df.Cr-\
    32.7*df.Ni+95.2*df.V+70.2*df.Ti+72*df.Al+64.5*df.Nb+\
    332*df.S+276*df.P-485*df.N+16.2*df.C*df.Mn+32.3*df.C*df.Si+\
    15.4*df.C*df.Cr+48*df.C*df.Ni+4.8*df.Mn*df.Ni+4.32*df.Si*df.Ni-\
    17.3*df.Si*df.Mo-18.6*df.Si*df.Ni+40.5*df.Mo*df.V+174*df.C*df.C+\
    2.46*df.Mn*df.Mn-6.86*df.Si*df.Si+0.322*df.Cr*df.Cr+9.9*df.Mo*df.Mo+\
    1.24*df.Ni*df.Ni-60.2*df.V*df.V-\
    900*df.B+5.57*df.W).round(0)
    return df

def calc_AC3_2(df):
    df[chem] = df[chem].fillna(0)
#     print(df[chem].describe())
    df['AC3_2'] = (912 - df.C*370-df.Mn*27.4+27.3*df.Si-6.35*df.Cr-\
    32.7*df.Ni+95.2*df.V+190*df.Ti+72*df.Al+64.5*df.Nb+\
    332*df.S+276*df.P+485*df.N+16.2*df.C*df.Mn+32.3*df.C*df.Si+\
    15.4*df.C*df.Cr+48*df.C*df.Ni+4.32*df.Si*df.Cr-17.3*df.Si*df.Mo-18.6*df.Si*df.Ni+\
    4.8*df.Mn*df.Ni+40.5*df.Mo*df.V+174*df.C*df.C+2.46*df.Mn*df.Mn-6.86*df.Si*df.Si+0.322*df.Cr*df.Cr+\
    9.9*df.Mo*df.Mo+1.24*df.Ni*df.Ni-60.2*df.V*df.V-
    900*df.B+5.57*df.W).round(0)
    return df

def calc_AC1_1(df):
    df[chem] = df[chem].fillna(0)
    df['AC1_1'] = (723-7.08*df.Mn+37.7*df.Si+18.1*df.Cr+44.2*df.Mo-8.95*df.Ni+50.1*df.V+21.7*df.Al+3.18*df.W+\
    297*df.S-830*df.N-11.5*df.C*df.Si-14*df.Mn*df.Si-3.1*df.Cr*df.Si-57.9*df.C*df.Mo-15.5*df.Mn*df.Mo-\
    5.28*df.C*df.Ni-6*df.Mn*df.Ni+6.77*df.Si*df.Ni-0.8*df.Cr*df.Ni-27.4*df.C*df.V+30.8*df.Mo*df.V-\
    0.84*df.Cr*df.Cr-3.46*df.Mo*df.Mo-0.46*df.Ni*df.Ni-28*df.V*df.V).round(0)
    return df

def calc_AC1_2(df):
    df[chem] = df[chem].fillna(0)
    df['AC1_2'] = (723-7.08*df.Mn+37.7*df.Si+18.1*df.Cr+44.2*df.Mo+8.95*df.Ni+50.1*df.V+21.7*df.Al+3.18*df.W+\
    297*df.S-830*df.N-11.5*df.C*df.Si-14*df.Mn*df.Si-3.1*df.Cr*df.Si-57.9*df.C*df.Mo-15.5*df.Mn*df.Mo-\
    5.28*df.C*df.Ni-6*df.Mn*df.Ni+6.77*df.Si*df.Ni-0.8*df.Cr*df.Ni-27.4*df.C*df.V+30.8*df.Mo*df.V-\
    0.84*df.Cr*df.Cr-3.46*df.Mo*df.Mo-0.46*df.Ni*df.Ni-28*df.V*df.V).round(0)
    return df
