import numpy as np
import os
import os.path
import pandas as pd
import torch
import xlrd
import openpyxl
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from utils.loader import FaultDataset
from torch.autograd import Variable


def minmax(data) -> pd.DataFrame:
    label_column = data['label']

    normalized_data = data.drop(columns=['label'])

    scaler = preprocessing.MinMaxScaler()
    normalized_data = scaler.fit_transform(normalized_data)

    columns = data.columns[:-1]
    normalized_df = pd.DataFrame(normalized_data, columns=columns)

    normalized_df['label'] = label_column.values

    return normalized_df

def split_data() -> pd.DataFrame:

    use_cols = ['TWE_set',
                'TEI', 'TWEI', 'TEO', 'TWEO', 'TCI', 'TWCI', 'TCO', 'TWCO',
                'TSI', 'TSO', 'TBI', 'TBO', 'Cond Tons', 'Cooling Tons', 'Shared Cond Tons',
                'Cond Energy Balance', 'Evap Tons', 'Shared Evap Tons', 'Building Tons',
                'Evap Energy Balance', 'kW', 'COP', 'kW/Ton', 'FWC', 'FWE', 'TEA', 'TCA',
                'TRE', 'PRE', 'TRC', 'PRC', 'TRC_sub', 'T_suc', 'Tsh_suc', 'TR_dis', 'Tsh_dis',
                'P_lift', 'Amps', 'RLA%', 'Heat Balance%', 'Tolerance%',
                'Unit Status', 'Active Fault', 'TO_sump', 'TO_feed', 'PO_feed', 'PO_net',
                'TWCD', 'TWED', 'VSS', 'VSL', 'VH', 'VM', 'VC', 'VE', 'VW', 'TWI', 'TWO',
                'THI', 'THO', 'FWW', 'FWH', 'FWB'
                ]

    normal = pd.read_excel('data_chiller/Benchmark Tests/normal.xls', usecols=use_cols)
    fwc40 = pd.read_excel('data_chiller/Reduced condenser water flow/fwc40.xls', usecols=use_cols)
    fwe40 = pd.read_excel('data_chiller/Reduced evaporator water flow/fwe40.xls', usecols=use_cols)
    rl40 = pd.read_excel('data_chiller/Refrigerant leak/rl40.xls', usecols=use_cols)
    ro40 = pd.read_excel('data_chiller/Refrigerant overcharge/ro40.xls', usecols=use_cols)
    eo50 = pd.read_excel('data_chiller/Excess oil/eo50.xls', usecols=use_cols)
    cf45 = pd.read_excel('data_chiller/Condenser fouling/cf45.xls', usecols=use_cols)
    nc5 = pd.read_excel('data_chiller/Non-condensables in refrigerant/nc5.xls', usecols=use_cols)

    fault_data_all = [normal,  fwe40, rl40, ro40, eo50, cf45, fwc40, nc5]

    for i, df in enumerate(fault_data_all):
        df['label'] = i

    fault_data = pd.concat(fault_data_all, ignore_index=True)

    fault_data = fault_data[(fault_data['TWE_set'] == 50)].reset_index(drop=True)

    print('Source data shape is {}'.format(fault_data.shape))
    fault_data = minmax(fault_data)

    fault_data_normal = fault_data[fault_data['label'] == 0]
    fault_data_fault = fault_data[fault_data['label'] != 0]
    # split_ratio = 0.7

    # train_data_normal = fault_data_normal.sample(frac=split_ratio, random_state=42)
    train_data_normal = fault_data_normal
    # test_data_normal = fault_data_normal.drop(train_data_normal.index)

    return train_data_normal,  fault_data_fault

def return_dataset_res():

    train_data_normal,  fault_data_fault = split_data()
    # s_fault_data, labeled_target, unlabeled_target, val_target = get_data(50, 40, 1)

    train_data_normal_i = FaultDataset(train_data_normal, test=False)
    # test_data_normal = FaultDataset(test_data_normal)
    fault_data_fault_i = FaultDataset(fault_data_fault, test=False)

    return train_data_normal_i, fault_data_fault_i, train_data_normal