import pandas as pd
import numpy as np
import re
#import matplotlib.dates as mdates
#from datetime import datetime

def transform_ddd(Fmonth):
    df=Fmonth.copy(deep=True)
    DDD=pd.read_csv("cleaned_cur_antibioDDDinfo.csv")
    
    # remove topical route that cannot be quantified.
    # to make sure it is clean remove all key words related to topical use
    comb_condA = ~(df['RX_ROUTE'].str.contains('EYE|NOSTRIL', na=False) | (df['RX_ROUTE'].isin(['Oral Application', 'Topical', 'Vaginal'])))
    df = df.loc[comb_condA & ~df['VOLUME_DOSE'].str.contains('application|ampoule|drop', na=False)]
    # remove inconsistent rows
    comb_condB = ~(df['ORDER_NAME'].str.contains('mL', na=False, regex=False)) & ~(df['VOLUME_DOSE'].str.contains('tab\\(s\\)|cap\\(s\\)', na=False, regex=False))
    comb_condC = ~(df['ORDER_NAME'].str.contains('Tablet|Capsule', na=False) & df['VOLUME_DOSE'].str.contains('mL'))
    df = df.loc[comb_condB | comb_condC]
    
    df['DOSE_UNIT']=None
    num_pattern = r'([\d,]+\.?\d*)\s*([A-Za-z/%()]+)?'
    df['ORDER*VOL']=None
    df['ORDER*VOL_UNIT']=None
    df['TIME_PER_DAY']=df['FREQUENCY']
    df['dosage_f24']=None
    
    # transform all convertable rows into ORDER*VOL
    for index, row in df.iterrows():
        if isinstance(row['DOSE'], str) and row['DOSE']:
            match_dose=re.match(num_pattern, row['DOSE'])
            if match_dose:
                num_str = match_dose.group(1)
                num_str=num_str.replace(',','')
                dose_num=float(num_str)
                unit = match_dose.group(2)
                if unit =="mg" or unit == 'mcg': # include this typo for now, impossible to include further typos
                    df.at[index,'DOSE']=dose_num
                    df.at[index,'DOSE_UNIT']='mg'
                elif unit == 'mg/kg':
                    if row['ACTUAL_WEIGHT']:
                        weight=row['ACTUAL_WEIGHT']
                        df.at[index, 'DOSE']=weight * dose_num
                        df.at[index, 'DOSE_UNIT']='mg'
                elif unit == 'g':
                    df.at[index,'DOSE']=dose_num * 1000
                    df.at[index, 'DOSE_UNIT']='mg'
                elif unit =='unit(s)':
                    df.at[index, 'DOSE']=dose_num
                    df.at[index, 'DOSE_UNIT']='unit'
            match_dose=row['DOSE'].split(' ')
            unit=match_dose[1]
            if unit == "International_Unit(s)":
                dose_str=match_dose[0].replace(',','')
                dose_num=float(dose_str)
                df.at[index, 'DOSE']=dose_num
                df.at[index, 'DOSE_UNIT']='unit'
            elif unit == "million_units":
                dose_str=match_dose[0].replace(',','')
                dose_num=float(dose_str)
                df.at[index, 'DOSE']=dose_num * 1000000
                df.at[index, 'DOSE_UNIT']='unit'
        if isinstance(row['VOLUME_DOSE'], str) and row['VOLUME_DOSE']:
            vol=row['VOLUME_DOSE']
            match_vol = vol.split(" ")
            numstr = match_vol[0]
            numstr = numstr.replace(',','')
            number_vol = float(numstr)
            unit=match_vol[1]
            order=row['ORDER_NAME']
            order=order.replace(" ",'')
            order=order.replace(',','')
            if unit == 'vial(s)':
                match_order = re.search(r'(\d+)g', order) 
                if match_order:
                    order_numstr=match_order.group(1)
                    order_number= float(order_numstr)
                    df.at[index, 'ORDER*VOL']= order_number * number_vol * 1000
                    df.at[index, 'ORDER*VOL_UNIT']='mg'
                else:
                    match_order= re.search(r'(\d+)mg', order)
                    if match_order:
                        order_numstr=match_order.group(1)
                        order_number = float(order_numstr)
                        df.at[index, 'ORDER*VOL']=order_number * number_vol
                        df.at[index,'ORDER*VOL_UNIT']='mg'
            elif unit == 'tab(s)' or unit =="cap(s)" or unit == "lozenge(s)" or unit == "sachet(s)":
                match_order = re.search(r'(\d+)mg', order)
                if match_order:
                    order_numstr=match_order.group(1)
                    order_number = float(order_numstr)
                    df.at[index, "ORDER*VOL"]=order_number * number_vol
                    df.at[index, 'ORDER*VOL_UNIT']='mg'
                else:
                    match_order = re.search(r'(\d+)g', order)
                    if match_order:
                        order_numstr=match_order.group(1)
                        order_number = float(order_numstr)
                        df.at[index, 'ORDER*VOL']=order_number * number_vol *1000
                        df.at[index,'ORDER*VOL_UNIT']='mg'
                    else:
                        match_order= re.search(r'(\d+)unit', order)
                        if match_order:
                            order_numstr=match_order.group(1)
                            order_number=float(order_numstr)
                            df.at[index, "ORDER*VOL"]=order_number * number_vol
                            df.at[index,'ORDER*VOL_UNIT']='unit'
                df.at[index, "RX_ROUTE"]='Oral'
            elif unit =="mL":
                mg_match=re.search(r'(\d+)mg', order)
                mL_match=re.search(r'(\d+)mL', order)
                g_match=re.search(r'(\d+)g', order)
                munit_match= re.search(r'(\d+)million_unit', order)
                unit_match= re.search(r'(\d+)unit', order)
                if mL_match:
                    mL_numstr=mL_match.group(1)
                    mL_num=float(mL_numstr)
                    if mg_match:
                        mg_numstr=mg_match.group(1)
                        mg_num=float(mg_numstr)
                        df.at[index, 'ORDER*VOL']= mg_num*number_vol/mL_num
                        df.at[index, 'ORDER*VOL_UNIT']='mg'
                    elif g_match: 
                        g_numstr=g_match.group(1)
                        g_num=float(g_numstr)
                        df.at[index, 'ORDER*VOL']= g_num*number_vol*1000/mL_num
                        df.at[index, 'ORDER*VOL_UNIT']='mg'
                    elif munit_match:
                        munit_nstr=munit_match.group(1)
                        munit=float(munit_nstr)
                        mL_numstr=mL_match.group(1)
                        mL_num=float(mL_numstr)
                        df.at[index,'ORDER*VOL']=munit*1000000*number_vol/mL_num
                        df.at[index, 'ORDER*VOL_UNIT']='unit'
                    elif unit_match:
                        unit_nstr=unit_match.group(1)
                        unit_num=float(unit_nstr)
                        df.at[index,'ORDER*VOL']=unit_num*number_vol/mL_num
                        df.at[index,'ORDER*VOL_UNIT']='unit'
                else:
                    if mg_match:
                        mg_numstr=mg_match.group(1)
                        mg_num=float(mg_numstr)
                        df.at[index, 'ORDER*VOL']= mg_num*number_vol
                        df.at[index, 'ORDER*VOL_UNIT']='mg'
                    elif g_match: 
                        g_numstr=g_match.group(1)
                        g_num=float(g_numstr)
                        df.at[index, 'ORDER*VOL']= g_num*number_vol*1000
                        df.at[index, 'ORDER*VOL_UNIT']='mg'
                    elif munit_match:
                        munit_nstr=munit_match.group(1)
                        munit=float(munit_nstr)
                        df.at[index,'ORDER*VOL']=munit*1000000*number_vol
                        df.at[index, 'ORDER*VOL_UNIT']='unit'
                    elif unit_match:
                        unit_nstr=unit_match.group(1)
                        unit_num=float(unit_nstr)
                        df.at[index,'ORDER*VOL']=unit_num*number_vol
                        df.at[index,'ORDER*VOL_UNIT']='unit'
    df['TIME_PER_DAY'] = df['TIME_PER_DAY'].replace(np.nan, None)                    
    for index, row in df.iterrows():
    # if DOSE not valid then copy ORDER*VOL to DOSE
        if isinstance(row['DOSE'], str) or pd.isna(row['DOSE']):
            df.at[index, 'DOSE']=row['ORDER*VOL']
            df.at[index, 'DOSE_UNIT']=row['ORDER*VOL_UNIT']
            row['DOSE'] = row['ORDER*VOL']
            exact_amt =row['DOSE']
        else: 
            exact_amt =row['DOSE']
            # Calculate the first 24 hour dose if antibiotic taken more than once per day
        if row['TIME_PER_DAY'] and row['DOSE']:
            freq = row['TIME_PER_DAY']
            if freq >=1:
                df.at[index, 'dosage_f24']=exact_amt*freq
    DDD_lst=DDD['name'].unique().tolist()
    # df=df[(df['RX_ROUTE'] != 'Miscellaneous')]
    
    StartStop = df[(-df['STOP_DTTM'].isna())].copy()
    StartStop = StartStop[(-StartStop['START_DTTM'].isna())]
    StartStop = StartStop[-StartStop['FREQUENCY'].isna()]
    StartStop = StartStop[-StartStop['DOSE'].isna()]
    
    dosage=df.copy(deep=True)
    dosage.sort_values(by=['MRN','ORDER_GENERIC','START_DTTM'], ascending=[True, True, True], inplace=True)
    mask=dosage.duplicated(subset=['MRN','ORDER_GENERIC'], keep='first')
    dosage_f24=dosage[~mask]
    dosage_f24=dosage_f24.dropna(subset=['EGFR','dosage_f24'])
    
    StartStop['DOT_per_ORDER']=None

# might need adjustment here depending on how Farros, Liz and Wilson format the datetime
# here it is using the original 4month only dataset so no problem currently
    StartStop['START_DTTM']=pd.to_datetime(StartStop['START_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M')
    StartStop['STOP_DTTM']=pd.to_datetime(StartStop['STOP_DTTM'], dayfirst= True, format = '%d/%m/%Y %H:%M')
    for index, row in StartStop.iterrows():
        start_date = row['START_DTTM'].date()
        stop_date = row['STOP_DTTM'].date()
        if stop_date == start_date:
            StartStop.at[index, 'DOT_per_ORDER']=1
        else:
            StartStop.at[index, 'DOT_per_ORDER']=(stop_date-start_date).days + 1
            
    # categorize all current route into O P and Inhal
    route_dict={'Oral':'O', 'Intravenous': 'P', 'IV Continuous Infusion': 'P', 'Nasogastric': 'O','Intraperitoneal': 'P', 'Nasojejunal':'O', 'IM': 'P', 'PEG': 'O', 'Jejunal': 'O', 'PEJ': 'O',
                'Nebulised': 'Inhal.solution', 'Haemodialysis': 'P', 'Intraosseous': 'P','IV Transfusion': 'P', 'Inhalation': 'Inhal.solution'}
    # all breathing route would be categorized as inhal solution for now. Would look into this later by searching key words
    StartStop['RX_ROUTE'] = StartStop['RX_ROUTE'].replace(route_dict)
    
    StartStop['total_DDD']= None
    StartStop['total_dosage']=None
    # test=[]
    for index, row in StartStop.iterrows():
        generic=row['ORDER_GENERIC'].lower()
        route=row['RX_ROUTE']
        exact_amt =row['DOSE']
        unit=row['DOSE_UNIT']
        freq = row['TIME_PER_DAY']
        length=row['DOT_per_ORDER']
        StartStop.at[index,'total_dosage']=exact_amt*freq*length
        if generic in DDD_lst:
            filterDDD=DDD[DDD['name']==generic]
            filter1DDD=filterDDD[filterDDD['Adm.R']==route]
            filter2DDD=filter1DDD[filter1DDD['U']==unit]
            if not filter2DDD.empty:
    #             if len(filter2DDD['DDD']) >1:
    #                 test.append(filter2DDD['name'].unique())
                typeDDD=float(filter2DDD['DDD'])
                StartStop.at[index,'DDD_per_day']=exact_amt*freq/typeDDD
                StartStop.at[index, 'total_DDD']=exact_amt*freq*length/typeDDD
        else:
            for item in DDD_lst:
                if generic in item or item in generic:
                    filterDDD=DDD[DDD['name']==item]
                    filter1DDD=filterDDD[filterDDD['Adm.R']==route]
                    filter2DDD=filter1DDD[filter1DDD['U']==unit]
                    if not filter2DDD.empty:
    #                     if len(filter2DDD['DDD']) >1:
    #                         test.append(filter2DDD['name'].unique())
                        typeDDD=float(filter2DDD['DDD'])
                        StartStop.at[index,'DDD_per_day']=exact_amt*freq/typeDDD
                        StartStop.at[index, 'total_DDD']=exact_amt*freq*length/typeDDD
    Fmonth['total_DDD']=None
    Fmonth['total_dosage']=None
    Fmonth['DOSE_UNIT']=None
    Fmonth['dosage_f24']=None
    for index, row in StartStop.iterrows():
        if not pd.isna(row['total_dosage']):
            Fmonth.at[index, 'total_dosage']=row['total_dosage']
        if not pd.isna(row['total_DDD']):
            Fmonth.at[index, 'total_DDD']=row['total_DDD']
    for index, row in dosage_f24.iterrows():
        if not pd.isna(row['dosage_f24']):
            Fmonth.at[index, 'dosage_f24']=row['dosage_f24']
    for index, row in df.iterrows():
        if not pd.isna(row['DOSE_UNIT']):
            Fmonth.at[index,'DOSE_UNIT']=row['DOSE_UNIT']
    Fmonth['EGFR']=Fmonth['EGFR'].replace({'>90': 91})
    return Fmonth