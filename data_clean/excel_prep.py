


import os
import pandas as pd
import numpy as np
import xlsxwriter
from config import *




def get_data(x_file,sheets,y_file):
    # Create a data column
    date = pd.date_range('2013-01-01', periods = 3652 , freq='D')
    ads = pd.DataFrame(date,columns=['Date'])
    
    #load the data 
    data_x = pd.ExcelFile(x_file)
    for i in range(0,sheets):
        sheet = data_x.parse(i)
        sheet.dropna(axis = 1 , how='all' ,inplace=True)
        sheet.fillna(value = '' ,axis = 0 , inplace = True)
        #print(i)
        
        #what if sheet positions are interchanged?
        if i == 0 :
            s = sheet
            s.columns = (s.iloc[1] + '_' + s.iloc[2]+ '_' + s.iloc[4])
            s = s[6:]
            s.reset_index(inplace=True, drop=True)
            s.rename(columns={s.columns[16]: "Date" }, inplace=True)
            s.rename(columns={s.columns[17]: "Handysize spot_BHSI Index_PX_LAST" }, inplace=True)
            s.rename(columns={ s.columns[0]: "Date" }, inplace=True)
            s.replace('',np.nan, regex=True, inplace = True)
            
            # Mergeing with main data column
            count = 2
            count_i = 0 
            df_len = len(list(s.columns))
            while count <= df_len:
                df_2 = s.iloc[: ,count_i : count]
                df_2['Date'] = pd.to_datetime(df_2['Date'])    
                if count == 2:
                    ads = pd.merge(ads, df_2, on = 'Date', how = 'left')
                else:
                    ads = pd.merge(ads, df_2, on = 'Date', how = 'left')
                count = count + 2
                count_i = count_i + 2
                
        elif i == 1: 
            cs = sheet
            cs.columns = (cs.iloc[0] + '_' + cs.iloc[5]+ '_' + cs.iloc[7])
            cs.rename(columns={ cs.columns[0]: "Date" },inplace = True)
            cs = cs[9:]
            cs.replace('',np.nan, regex=True, inplace = True)
            
        elif i == 2 :
            pm = sheet
            pm.columns = (pm.iloc[2] + '_' + pm.iloc[7]+ '_' + pm.iloc[9])
            pm.rename(columns={ pm.columns[0]: "Date" },inplace = True)
            pm = pm[11:]
            pm.replace('',np.nan, regex=True, inplace = True)
            
        elif i == 3 : 
            sm = sheet
            sm.columns = (sm.iloc[2] + '_' + sm.iloc[7]+ '_' + sm.iloc[9])
            sm = sm[11:]
            sm.rename(columns={ sm.columns[0]: "Date" },inplace = True)
            sm.replace('',np.nan, regex=True, inplace = True)
           
        elif i == 4 : 
            hm = sheet
            hm = hm.loc[1:]
            hm.rename(columns={ hm.columns[0]: "Date" },inplace = True)
            hm.replace('',np.nan, regex=True, inplace = True)
        
        else : 
            print("Sheet No. ",i," has no cleaning process setup ")
        
    # Mergeing all the files to create main df    

    df= ads.merge(cs, on ='Date', how = 'left').merge(pm,on='Date',how = 'left').merge(sm,on='Date',how = 'left').merge(hm,on='Date',how =         'left')
    df_x = df.groupby(pd.Grouper(key='Date', axis=0, freq='M')).mean().reset_index()

    df_y = pd.read_excel(y_file)
    df_y['Date'] = df_y['Date'] + pd.offsets.MonthEnd(n=0)

    all_feature_df = df_x.merge(df_y, on = 'Date', how = 'left' )

    all_feature_df.fillna(0, inplace = True)
    all_feature_df = all_feature_df[(all_feature_df['Date'] >= ANALYSIS_START_DATE)]

    
    all_feature_df.set_index('Date',inplace = True)
    all_feature_df.index = pd.to_datetime(all_feature_df.index)

    return all_feature_df
    





def Output(y_var,final_output) :
    path = y_var
    print(path)
    master_0 = pd.DataFrame()
    master_1 = pd.DataFrame()
    master_2 = pd.DataFrame()
    master_3 = pd.DataFrame()

    for name in os.listdir(path):
        print(name)
        for filename in os.listdir(path+name):   
            if filename == 'FINAL_FORECAST.xlsx' :
                dir_to_move = os.path.join(path,name,filename)
                df_k = pd.ExcelFile(dir_to_move)
                sheetname = df_k.sheet_names

                for i in range(0,len(pd.read_excel(dir_to_move,sheet_name=None))):
                        sheet = df_k.parse(i)
                        sheet['country/port'] = name
                        sheet['Start Date'] = ANALYSIS_START_DATE
                        sheet['End Date'] = TAKE_ACTUAL_TILL
                        sheet['Frequency'] = FREQ
                        sheet['Forecasting period'] = ADD_LAG
                        sheet.rename(columns={sheet.columns[0]: "Forecast_Date" }, inplace=True)

                        if i == 0 :
                            sheet_0 = sheet
                        elif i == 1:
                            sheet_1 = sheet 
                        elif i == 2:
                            sheet_2 = sheet 
                        elif i == 3 : 
                            sheet_3 = sheet
                master_0 = pd.concat([master_0, sheet_0])
                master_1 = pd.concat([master_1, sheet_1])
                master_2 = pd.concat([master_2, sheet_2])
                master_3 = pd.concat([master_3, sheet_3])



    writer = pd.ExcelWriter(final_output, engine = 'xlsxwriter')
    master_0.to_excel(writer, sheet_name = str(sheetname[0]))
    master_1.to_excel(writer, sheet_name = str(sheetname[1]))
    master_2.to_excel(writer, sheet_name = str(sheetname[2]))
    master_3.to_excel(writer, sheet_name = str(sheetname[3]))
    writer.save()
    writer.close()
    print("Output ready!")
    
    
    
    
    
    

