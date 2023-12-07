import pandas as pd
from datetime import datetime
from transform_dot import transform_dot

def dot_ddd_dasc(DDD, filepath = './orders_data_clean/clean.csv'):
    df = pd.read_csv(filepath)

    print("TRANSFORM DOT START")

    df = transform_dot(df, filepath = 'ASC_SCORES.csv')
    
    print("TRANSFORM DOT END")

    print("MERGE START")

    final_df = pd.read_csv('./orders_data_clean/final_df.csv')
    final_df = final_df.drop(columns = ['DAYS_OF_THERAPY','ASC_SCORE','dASC'])
    final_df['ADMIT_DATE'] = pd.to_datetime(final_df['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y').dt.date
    #final_df['START_DTTM'] = pd.to_datetime(final_df['START_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M')
    DDD['ADMIT_DATE'] = pd.to_datetime(DDD['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.date
    #DDD['ADMIT_DATE'] = pd.to_datetime(DDD['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y').dt.strftime('%d/%m/%Y')# truncate time
    DDD = pd.concat([DDD,final_df])
    DDD['START_DTTM'] = pd.to_datetime(DDD['START_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M')#.dt.strftime('%d/%m/%Y %H:%M') # convert to datetime
    #DDD['ADMIT_DATE'] = pd.to_datetime(DDD['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.date

    DDD.reset_index(drop=True, inplace=True)
    mask = DDD['START_DTTM'].notna() # mask to exclude rows with empty start_dttm values
    earliest_start = DDD[mask].groupby(['MRN', 'ADMIT_DATE', 'ORDER_GENERIC'])['START_DTTM'].idxmin() # partition by mrn, admit_date, order_generic and index earliest start_dttm
    merged_df = DDD.loc[earliest_start].merge(df, on=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC'], how='left') # left join on mrn, admit_date, order_generic and where start_dttm is earliest
    merged_df.sort_values(by=['MRN', 'ADMIT_DATE', 'ORDER_GENERIC', 'START_DTTM'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True) # reset index
    merged_df = merged_df.drop(columns = ['AMS_INDICATION_y', 'MEDICAL_SERVICE_y'])
    merged_df = merged_df.rename(columns = {'MEDICAL_SERVICE_x':'MEDICAL_SERVICE', 'AMS_INDICATION_x':'AMS_INDICATION'})
    merged_df['ADMIT_DATE'] = pd.to_datetime(merged_df['ADMIT_DATE'], dayfirst= True, format = '%d/%m/%Y').dt.strftime('%d/%m/%Y')

    to_date = ['ORDER_PLACED_DATE','DISCHARGE_DATE','START_DTTM','STOP_DTTM']
    for i in to_date:
        merged_df[i] = pd.to_datetime(merged_df[i], dayfirst= True, format = '%d/%m/%Y %H:%M').dt.strftime('%d/%m/%Y %H:%M')
    merged_df = merged_df.drop_duplicates()

    print("MERGE END")

    return merged_df