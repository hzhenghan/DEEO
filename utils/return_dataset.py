import numpy as np
import os
import os.path
import pandas as pd
import torch
import xlrd
import openpyxl
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from data_loader.loader import FaultDataset
from torch.autograd import Variable
from collections import Counter

def minmax(data) -> pd.DataFrame:
    label_column = data['label']

    normalized_data = data.drop(columns=['label'])

    scaler = preprocessing.MinMaxScaler()
    normalized_data = scaler.fit_transform(normalized_data)

    columns = data.columns[:-1]
    normalized_df = pd.DataFrame(normalized_data, columns=columns)

    normalized_df['label'] = label_column.values

    return normalized_df

def split_data(path, res) -> pd.DataFrame:
    fault_data = pd.read_csv(path)
    if not res:
        fault_data = minmax(fault_data)
    fault_data = fault_data.reset_index(drop=True)

    return fault_data


def return_dataset(source_path, target_path, evaluation_path, return_id=True, batch_size=8, res=False, balanced=False):

    source_data = split_data(source_path, res=res)
    target_data = split_data(target_path, res=res)
    evaluation_data = split_data(evaluation_path, res=res)

    source_folder = FaultDataset(source_data, return_id=return_id)
    target_folder_train = FaultDataset(target_data, return_id=return_id)
    eval_folder_test = FaultDataset(evaluation_data)

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True)

    else:
        source_loader = torch.utils.data.DataLoader(source_folder,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_folder_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(eval_folder_test, batch_size=batch_size,shuffle=False)

    return source_loader, target_loader, test_loader, target_folder_train

