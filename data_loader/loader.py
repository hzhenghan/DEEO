import numpy as np
import os
import os.path
import pandas as pd
import torch.utils.data
import xlrd
import openpyxl
from sklearn import preprocessing


class FaultDataset(torch.utils.data.Dataset):
    def __init__(self, data, fea_cols=['TWE_set', 'TEI', 'TWEI', 'TEO', 'TWEO', 'TCI', 'TWCI', 'TCO', 'TWCO',
                'TSI', 'TSO', 'TBI', 'TBO', 'Cond Tons', 'Cooling Tons', 'Shared Cond Tons',
                'Cond Energy Balance', 'Evap Tons', 'Shared Evap Tons', 'Building Tons',
                'Evap Energy Balance', 'kW', 'COP', 'kW/Ton', 'FWC', 'FWE', 'TEA', 'TCA',
                'TRE', 'PRE', 'TRC', 'PRC', 'TRC_sub', 'T_suc', 'Tsh_suc', 'TR_dis', 'Tsh_dis',
                'P_lift', 'Amps', 'RLA%',  'Tolerance%','Heat Balance%',
                'Unit Status', 'Active Fault', 'TO_sump', 'TO_feed', 'PO_feed', 'PO_net',
                'TWCD', 'TWED', 'VSS', 'VSL', 'VH', 'VM', 'VC', 'VE', 'VW', 'TWI', 'TWO',
                'THI', 'THO', 'FWW', 'FWH', 'FWB'
                ],
                 return_id=False
                 ):

        self.fea_data = data[fea_cols].values
        self.labels = data['label'].values
        self.return_id = return_id


    def __getitem__(self, index):
        # path = os.path.join(self.root, self.imgs[index])
        # target = self.labels[index]

        data = self.fea_data[index]
        target = self.labels[index]

        if self.return_id:
            return data, target, index
        else:
            return data, target

    def __len__(self):
        return len(self.fea_data)

    def process(self):
        pass
