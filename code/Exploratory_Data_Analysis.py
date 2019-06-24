##################################################
#                Import module                   # 
##################################################
import pandas as pd
import numpy as np 

#Statistics Module
import pylab 
from scipy import stats
from scipy.stats import chi2
import statsmodels.api as sm 
from collections import Counter
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from scipy.stats import anderson, shapiro

# Check Normalization of the Data
from statsmodels.graphics.gofplots import qqplot

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

#########################################################
#                PREVIEW DATASET FUNCTION               #
#########################################################

def preview_df(df, options = None):
    '''
    Preview dataset is one of exploratory data analysis part, which is 
    we will know what the dataset is. 

    Paramaters :
    ------------
    df      :   object, DataFrame
            Dataset that will be used  
    option  :   Optional(default = 'top record data')
            1. top_record data  : Showing top record data(default = 10)
            2. shape_data       : showing how many rows and column of dataset
            3. info_data        : showing how many columns that includes missing value
                                  and knowing what the data type is of each column. 
    '''
    #default option is Top record 
    if options == None :
        options = 'top_record'
    
    if options == 'top_record':
        print('=>> Top 10 Record Data : ','\n')
        df = df.head(10)
        return df 

    if options == 'shape_data':
        print('=>> Data shape : ','\n')
        df = df.shape
        return df

    if options == 'info_data':
        print('=>> Data Info : ','\n')
        df = df.info()
        return df   

  
#########################################################
#             STATSITCS DESCRIPTIVE FUNCTION            #
#########################################################

# Statistics Descriptive
def describe_stats(df,column=None):
    '''
    Parameter :
    -----------
    * df  : Dataframe Name
    * col : Columns Name    
    '''
    if column is None :
        df = df.describe()
    else:
        df = df[column].describe()
        
    return df


#########################################################
#                 TYPE OF DATA 1 FUNCTION               #
#########################################################
# Seperate the features based on types of data
# function 1
def check_dtypes_1(df):
    '''
    Parameters :
    ------------
    * df : Dataframe name 
    
    Step :
    ------
    * 1. Do iteration for each feature to define which one categorical and nummerical feature. 
    * 2. Columns in dataframe will be seperated based on the dtypes
    * 3. All of the column will be entered to the list that have been created

    result :
    --------
    The result will be formed as dataframe
    '''
    # Make a list for both of the data type 
    categorical_list = []
    numerical_list = []
    
    #Looping 
    for col in df.columns.tolist():
        if df[col].dtype=='object':
            categorical_list.append(col)
        else:
            numerical_list.append(col)
    
    #make dataframe that have two feature, that is categorical and numerical feature
    categorical = pd.Series(categorical_list, name='Categorical Feature')
    numerical = pd.Series(numerical_list, name='Numerical Feature')
    df_dtypes = pd.concat([categorical,numerical], axis=1)
    
    return df_dtypes


#########################################################
#                 TYPE OF DATA 2 FUNCTION               #
#########################################################

#function 2
def check_dtypes_2(df):
    '''
    Parameters :
    ------------
    * df : Dataframe name 
    
    Step :
    * 1. Do iteration for each feature to define which one categorical and nummerical feature. 
    * 2. Columns in dataframe will be seperated based on the dtypes
    * 3. All of the column will be entered to the list that have been created

    result:
    The result will be formed as dataframe
    '''
    # Seperate the features based on types of data
    float_type = []
    int_type = []
    object_type = []
    
    #Looping 
    for col in df.columns.tolist():
        if df[col].dtype =='float64':
            float_type.append(col)
        elif df[col].dtype =='int64':
            int_type.append(col)
        else:
            object_type.append(col)
    
    #make dataframe that have two feature, that is categorical and numerical feature 
    float = pd.Series(float_type, name='float type')
    int = pd.Series(int_type, name='int type')
    object = pd.Series(object_type, name='object type')
    df_dtypes = pd.concat([float,int,object], axis=1)
    
    return df_dtypes 

#########################################################
#                 UNIQUE COLUMNS FUNCTION               #
#########################################################
# How many unique value
def unique_columns(df):
    '''
    Parameter 
    ---------
    df : array
        the data frame that will be checked the unique the entities 
    '''
    for col in df.columns:
        if df[col].dtypes == 'object':
            unique_cat = len(df[col].unique())
            print("Feature '{col}' has {unique_cat} unique categories".format(col=col, unique_cat=unique_cat))


#########################################################
#             CHECKING MISSING VALUE FUNCTION           #
#########################################################
def missing_value(df):
    '''
    Documentation :
    --------------
    * df : Dataframe Name
    '''
    #count the number of missing value 
    total = df.isnull().sum()
    percent = round(df.isnull().sum()/len(df)*100,2)
    missing  = pd.concat([total, percent], axis=1, keys=['Total_Missing', 'Percent(%)'])
    missing.sort_values(by='Total_Missing', ascending=False, inplace=True)
    
    return print(missing.head(20))

#########################################################
#               DROP MISSING VALUE FUNCTION             #
#########################################################
#module 
def Drop_Missing(df, threshold = None):
    '''
    Documentation   :
    ---------------
    df              :  Object, Dataframe
                    the dataframe which want to dropped     
    threshold       :   float, default (0.75)
                    the number of threshold was determined by user 
    '''
    #default number of threshold 
    if threshold == None : 
        threshold = 0.75

    # Define variable that we need 
    threshold = threshold
    size_df = df.shape[0]

    # Define Column list that will we removed
    dropcol = []

    #looping to take the number of null of every feature 
    for col in df.columns :
        if (df[col].isnull().sum()/size_df >= threshold):
            dropcol.append(col)
    df = df.drop(dropcol, axis =1)

    return df

#########################################################
#                LIST DATA TYPES FUNCTION               #
#########################################################
#function
def list_dtypes(df):
    categorical_list = []
    numerical_list = []
    for col in df.columns.tolist():
        if df[col].dtype=='object':
            categorical_list.append(col)
        else:
            numerical_list.append(col)
    print('Number of categorical features:', str(len(categorical_list)))
    print('Number of numerical features:', str(len(numerical_list)))

    return categorical_list, numerical_list

#########################################################
#             FILLING MISSING VALUE FUNCTION            #
#########################################################
# Filling Missing value 
def fill_missing(df, feature_list = None , vartype = None ):
    ''' 
    Parameters :
    ------------
    df              : dataframe name
    feature_list    : feature list is the set of numerical or categorical features 
                      that have been seperated before
    vartype         : variable type : continuos or categorical, default (numerical)
                        (0) numerical   : variable type continuos/numerical
                        (1) categorical : variable type categorical
     
    
    Note :
    ------
    > if numerical variable will be filled by median 
    > if categorical variabe will filled by modus
    > if have been made variebles based on the dtypes list before, 
      insert it into feature list in the function.     

    Example :
    ---------
    # 1. Define feature that will be filled in 
      num_feature = numeric_list
      
    # 2. Input Dataframe
      dataframe = df
      
    # 3. Vartype
      var_type = 0
      
    # 4. Filling Value
      Fill_missing(dataframe, num_feature, var_type)
    '''
    #default vartype 
    if vartype == None :
        vartype = 'numerical'

    # filling numerical data with median 
    if vartype == 'numerical' :
        for col in feature_list:
            df[col] = df[col].fillna(df[col].median(), inplace = True)
    
    # filling categorical data with modus  
    if vartype == 'categorical' :
        for col in feature_list:
            df[col] = df[col].fillna(df[col].mode().iloc[0],inplace =True)


#########################################################
#                  NORMALITY TEST FUNCTION              #
#########################################################
# Checking Normality plot 
def normality_test(df, methods = None , numerical_columns = None, alpha = None):
    if methods == None:
        methods = 'univariate'
    
    if alpha == None :
        alpha = 0.05
    
    if methods == 'univariate':
        _, p = shapiro(df[numerical_columns])
        print('P_value = %.2f' % (p))
        alpha = alpha 
        if p >= alpha :
            print('Sample looks Gaussian (fail to reject H0) : normal distribution')
        else:
            print('Sample does not look Gaussian (reject H0) : not normally distributed')
        
        #qqplot graph 
        plt.figure(figsize=(10,6))
        stats.probplot(df[numerical_columns], dist = "norm", plot=pylab)
        pylab.show()

    if methods == 'multivariate':
        columns = numerical_columns 
        result = []
        #for loop 
        for col in numerical_columns : 
            _, p = shapiro(df[col])
            result.append(p)
        
        result = np.array(result)
        column = np.array(columns)
        table  = pd.DataFrame({'feature':column, 'p_value':result})
        table['alpha'] = 0.05
        table['decision'] = np.where(table['p_value'] >= 0.05,'normal distribution', 'not normally distributed')
                    
        return table 


#########################################################
#               STRONG CORRELATION FUNCTION             #
#########################################################
#Function Strong Correlation 
def strong_corr(df, ycol=None):
    '''
    Documentation :
    --------------
    Df   : Dataset name
    ycol : The target column of the dataframe
    '''
    df_num_corr = df.corr()[ycol][:-1] # -1 because the latest row is SalePrice
    strong_corr = df_num_corr[abs(df_num_corr) > 0.6].sort_values(ascending=False)
    corr = pd.Series(strong_corr, name='Storng Corr')
    df = pd.concat([corr], axis = 1) 
    
    return df


#########################################################
#              UNIVARIATE OUTLIERS DETECTION            #
#########################################################
def univariate_detect_outliers(df,col = None):

    #calulate q25, q75, and iqr 
    q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    iqr = q75 - q25
    
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    
    # identify outliers
    outliers = [x for x in df[col] if x < lower or x > upper]
    proportion = round(len(outliers)/len(df)*100,1)
    
    # remove outliers
    outliers_removed = [x for x in df[col] if x >= lower and x <= upper]

    #print function 
    print('Percentiles: 25th=%.2f, 75th=%.2f, IQR=%.2f' % (q25, q75, iqr))
    print('Identified outliers: %d' % len(outliers))
    print('Outliers proportion: %d' % proportion, '%')
    print('Non-outlier observations: %d' % len(outliers_removed))

#########################################################
#            DETECTING OUTLIERS VALUE FUNCTION          #
#########################################################
def multivariate_detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers 


#########################################################
#                HANDLING OUTLIERS FUNCTION             #
#########################################################
def outlier_handling (dataframe, num_cols, method=None, upper=None, lower=None):
    """
    Change the value of outliers based on the chosen method. Currently there are
    two outlier handling methods, which are outer boundary and HDBSCAN method.
    Remember, this function can only be applied to numerical columns only.
    
    Parameters
    ----------
    
    dataframe:    object, DataFrame
        Dataframe which its outliers values are going to be changed.

    num_cols:     array, string
        Name of the columns that are going to be changed.

    method:       string, default 'outer_boundary'
        Outlier handling method which are going to be used.
        There are two methods, outer_boundary and hdbscan.
        outer_boundary: replace the value of outer value with the value of outer boundary
        hdbscan       : Clustering Case, replace the value of outer value with median of cluster
        Note:
        if hdbscan method is being used, upper and lower parameter is unused.
        
    upper:        boolean, default True
        Decide whether going to drag upper outliers to upper bound or not.
        
    lower:        boolean, default True
        Decide whether going to drag lower outliers to lower bound or not.

    Returns
    -------
    
    Dataframe:
        The function returns processed dataframe.

    """
    # Parameter
    if method is None:
        method = 'outer_boundary'

    if upper is None:
        upper = True
        
    if lower is None:
        lower = True

    assert type(method) is str, "method is not a string: %r" % method
    assert type(upper) is bool, "upper is not a boolean: %r" % upper
    assert type(lower) is bool, "fill_method is not a boolean: %r" % lower

    updated_df = dataframe

    for column in num_cols:
        updated_df[column] = pd.to_numeric(updated_df[column], errors='coerce')

    # Case of using outer boundary method
    if method is 'outer_boundary':
        # Define The Calculation of Bounds
        def boundary_values (dataframe, column):
            q1 = dataframe[column].quantile(q=0.25)
            q3 = dataframe[column].quantile(q=0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            return lower_bound, upper_bound

        # Drag The Upper and Lower Outlier to The Upper Bound
        if (upper is True) & (lower is True):
            for column in num_cols:
                lower_bound, upper_bound = boundary_values(updated_df, column)
                updated_df.loc[updated_df[column] > upper_bound, column] = upper_bound
                updated_df.loc[updated_df[column] < lower_bound, column] = lower_bound
            print('\nOutlier Handling: COMPLETED')
            print('Handled outliers: Upper Bound and Lower Bound')

        # Drag The Upper Outlier to The Upper Bound
        elif (upper is True) & (lower is False):
            for column in num_cols:
                lower_bound, upper_bound = boundary_values(updated_df, column)
                updated_df.loc[updated_df[column] > upper_bound, column] = upper_bound
            print('\nOutlier Handling: COMPLETED')
            print('Handled outliers: Upper Bound')

        # Drag The Lower Outlier to The Upper Bound
        elif (upper is False) & (lower is True):
            for column in num_cols:
                lower_bound, upper_bound = boundary_values(updated_df, column)
                updated_df.loc[updated_df[column] < lower_bound, column] = lower_bound
            print('\nOutlier Handling: COMPLETED')
            print('Handled outliers: Lower Bound')


        # Nothing is changed
        elif (upper is False) & (lower is False):
            print('No outlier is changed here')

        # False input
        else:
            print('\nPlease input a proper boolean type (True or False). No outlier is changed here')

    # Case of using hdbscan (GLOSH) method
    #elif method is 'hdbscan':
        #clusterer = HDBSCAN(min_cluster_size=15).fit(updated_df[num_cols])
        #outliers_score = clusterer.outlier_scores_
        #threshold = pd.Series(outliers_score).quantile(0.9)
        #outliers = np.where(outliers_score > threshold)[0]
        #for column in num_cols:
            #for score in outliers:
                #updated_df.loc[updated_df[column] == score, column] = updated_df[column].mode()
        #print('\nOutlier Handling: COMPLETED')
        #print('Handled outliers: HDBSCAN Method')

    return updated_df


#########################################################
#              FEATURE ENGINEERING FUNCTION             #
#########################################################

#def aggregation(df, column_list = None) :
   # df.groupby('')