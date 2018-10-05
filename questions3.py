import json
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.DataFrame({'value(t)':range(100)})
df=df[:-2]

df['value(t+2)'] = df['value(t)']+2

print(df)

