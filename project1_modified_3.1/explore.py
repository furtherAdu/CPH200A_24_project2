import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataset import NLST
from csv import DictReader

#nlst_data = NLST()
#nlst_data.setup()
#test_data = nlst_data.test.dataset  # Access the test dataset
#test_labels = [sample["y"] for sample in test_data]
#test_pid = [sample["pid"] for sample in test_data]
#pd.DataFrame(test_labels, test_pid).to_csv('nlst_test_labels.csv')

#prsn_data = pd.read_csv('/scratch/users/rbhalerao/project1/nlst_564_prsn_20191001.csv')
#prsn_data_test_filt = prsn_data[prsn_data["pid"] == test_pid]
#prsn_data_test_filt['lung_cancer'] = test_labels
#prsn_data_test_filt.to_csv('nlst_test_labels.csv')

nlst_metadata = pd.read_csv('/Users/rbhalerao/Desktop/project1_modified/nlst_564_prsn_20191001.csv')
nlst_set = pd.read_csv('/Users/rbhalerao/Desktop/project1_modified/nlst_test_labels.csv')
nlst_set.dropna()
nlst_agg = nlst_metadata[nlst_metadata['pid'].isin(nlst_set['pid'])]
nlst_agg['label'] = nlst_set['label'].apply(int)
nlst_agg['label'].replace('', '0') 
nlst_agg['label'].fillna('0')

#nan_rows = nlst_agg[nlst_agg['label'].isna()]
#print(nan_rows)

# Find rows where 'label' is an empty string
#empty_rows = nlst_agg[nlst_agg['label'] == '']
#print(empty_rows)

nlst_agg = nlst_agg.dropna(subset=['label'])
nan_rows = nlst_agg[nlst_agg['label'].isna()]

nlst_agg.to_csv('nlst_agg.csv')
#print(type(nlst_agg['label'].values[0]))
#print(int(nlst_agg['label'].values[0]))
nlst_agg = DictReader(open('/Users/rbhalerao/Desktop/project1_modified/nlst_agg.csv',"r"))
nlst_Y = np.array([(int(float(r["label"]))) for r in nlst_agg])
print(type(nlst_Y[0]))