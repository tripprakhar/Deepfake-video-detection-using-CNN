# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:11:36 2021

@author: tripprakhar
"""

import os
from shutil import copy2
import pandas as pd

metadata_df = pd.read_json('./train/metadata.json')
metadata_df.head()
for lab in list(metadata_df):
    src=os.path.join('./train',lab)
    if metadata_df[lab]['label']=='FAKE':
        dst=os.path.join('./fake')
    else:
        dst=os.path.join('./real')
    copy2(src,dst)  