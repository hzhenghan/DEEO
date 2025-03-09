import numpy as np
import os
import os.path
import pandas as pd
import torch
import xlrd
import openpyxl
from sklearn import preprocessing


class split_data(object):
    def __init__(self,
                 d1: int = None,
                 d2: int = None,
                 d1_path_res: str = None,
                 d2_path_res: str  =None
                 ):
        def minmax(data) -> pd.DataFrame:
            label_column = data['label']
            normalized_data = data.drop(columns=['label'])
            scaler = preprocessing.MinMaxScaler()
            normalized_data = scaler.fit_transform(normalized_data)
            columns = data.columns[:-1]
            normalized_df = pd.DataFrame(normalized_data, columns=columns)
            normalized_df['label'] = label_column.values

            return normalized_df

        self.minmax = minmax()
        self.d1 = d1
        self.d2 = d2
        self.d1_path = d1_path_res
        self.d2_path = d2_path_res

    def split_original_data(self):
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

        fault_data_all = [normal, fwc40, fwe40, rl40, ro40, eo50, cf45, nc5]

        for i, df in enumerate(fault_data_all):
            df['label'] = i

        fault_data = pd.concat(fault_data_all, ignore_index=True)
        fault_data = self.minmax(fault_data)
        s_fault_data = fault_data[(fault_data['TWE_set'] == self.d1)]
        t_fault_data = fault_data[(fault_data['TWE_set'] == self.d2)]

        return s_fault_data, t_fault_data

    def split_res_oda(self):
        s_fault_data = pd.read_csv(self.d1_path)
        t_fault_data = pd.read_csv(self.d2_path)

        s_fault_data = s_fault_data[s_fault_data['label'].isin([0, 1, 2, 3, 4])]
        t_fault_data = t_fault_data
        t_fault_data['label'] = t_fault_data['label'].apply(lambda x: 5 if x not in [0, 1, 2, 3, 4] else x)

        return s_fault_data, t_fault_data

    def split_data_pda(self):
        s_fault_data = pd.read_csv(self.d1_path)
        t_fault_data = pd.read_csv(self.d2_path)

        t_fault_data = t_fault_data[t_fault_data['label'].isin([0, 1, 2, 3, 4, 5])]

        return s_fault_data, t_fault_data

    def split_data_opda(self):
        s_fault_data = pd.read_csv(self.d1_path)
        t_fault_data = pd.read_csv(self.d2_path)

        s_fault_data = s_fault_data[s_fault_data['label'].isin([0, 1, 2, 3, 4, 5, 6])]
        t_fault_data['label'] = t_fault_data['label'].apply(lambda x: 7 if x not in [0, 1, 2, 3, 4] else x)

        return s_fault_data, t_fault_data


if __name__ == '__main__':
    split_data = split_data(40, 50,
                            'data_chiller/data_res_feature/fault_data_40.csv',
                            'data_chiller/data_res_feature/fault_data_50.csv')

    s_fault_data, t_fault_data = split_data.split_res_oda()

    s_fault_data.to_csv('data_chiller/PDA/source_40_pda.csv')
    t_fault_data.to_csv('data_chiller/PDA/target_50_pda.csv')