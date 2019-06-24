#import module 
import pandas as pd 
import numpy as np 

#####################################################################
###                    DROP MULTYPLE COLUMNS                      ###
#####################################################################
def drop_multiple_col(df, columns_list):
    '''
    parameter :
    -----------
    df           :   Dataframe
                  Dataframe that used
    columns_list :   list, string s
                  columns list that will be removed 

    exp :
    - starting from make colum list :
      column_list = ['A','B','C','D']
    - use df that wanna entered to the function
    drop_multiple_col(df, column_list)
    '''
    df.drop(columns_list, axis=1, inplace=True)
    return df


#####################################################################
###                        CHANGE DTYPES                          ###
#####################################################################
def change_dtypes(df, col_int, col_float): 
    '''
    parameter :
    -----------
    df           :   Dataframe
                  Dataframe that used
    col_int      :   string 
                  integer column from the dataframe
    col_float    :   string
                  float column from the dataframe
 
    '''
    df[col_int] = df[col_int].astype('int32')
    df[col_float] = df[col_float].astype('float32')



#####################################################################
###       Convert categorical variable to numerical variable      ###
#####################################################################
def convert_cat2num(df):
    '''
    parameter : 
    -----------
    df              :  Dataframe
                     The dataframe that used    
    '''
    # Convert categorical variable to numerical variable
    num_encode = {'col_1' : {'YES':1, 'NO':0},
                  'col_2'  : {'WON':1, 'LOSE':0, 'DRAW':0}}  
    df.replace(num_encode, inplace=True)



#########################################################
#             CHECKING MISSING VALUE FUNCTION           #
#########################################################
def missing_value(df):
    '''
    parameter :
    -----------
    df          : Array
                 Dataframme that will be checked 
    '''
    #count the number of missing value 
    total = df.isnull().sum()
    percent = round(df.isnull().sum()/len(df)*100,2)
    missing  = pd.concat([total, percent], axis=1, keys=['Total_Missing', 'Percent(%)'])
    missing.sort_values(by='Total_Missing', ascending=False, inplace=True)
    
    print(missing.head(20))


#########################################################
#     remove white space at the beginning of string     #
#########################################################
def remove_col_white_space(df,col):
    '''
    parameter :
    -----------
    df        :  Array 
                Dataframe that will be used 
    col       : string 
                Column
    '''
    df[col] = df[col].str.lstrip()



#########################################################
# Concatenate two columns with strings (with condition) #
#########################################################
def concat_col_str_condition(df):
    '''
    parameter :
    -----------
    df          : Array
                 Dataframme that will be checked 
    '''
    # concat 2 columns with strings if the last 3 letters of the first column are 'pil'
    mask = df['col_1'].str.endswith('pil', na=False)
    col_new = df[mask]['col_1'] + df[mask]['col_2']
    col_new.replace('pil', ' ', regex=True, inplace=True) # replace the 'pil' with emtpy space

#########################################################
# Concatenate two columns with strings (with condition) #
#########################################################
def convert_str_datetime(df): 
    '''
    parameter :
    -----------
    df          : Array
                 Dataframme that will be checked 

    AIM    -> Convert datetime(String) to datetime(format we want)

    INPUT  -> df
    
    OUTPUT -> updated df with new datetime format 
    ------
    '''
    df.insert(loc=2, column='timestamp', value=pd.to_datetime(df.transdate, format='%Y-%m-%d %H:%M:%S.%f'))