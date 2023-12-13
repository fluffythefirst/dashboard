import pandas as pd
from cleaning_function import *

df = pd.read_csv('abx_MAY_DUMMY.csv')
output = cleaning(df)
output.to_csv('may_clean.csv', index = False, quoting = 1)