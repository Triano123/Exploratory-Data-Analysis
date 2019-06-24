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

#Check Normalization of the Data
from statsmodels.graphics.gofplots import qqplot

#visualization 
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt 


#########################################################
#                    DATA EXPLORATION                   #
#########################################################
#-----------------------------------------------#
#                 Preview Dataset               #
#-----------------------------------------------#
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

#-----------------------------------------------#
#            Descriptive Statistics             #
#-----------------------------------------------#
def describe_stats(df,column=None):
    '''
    Parameters :
    -----------
    * df  : Dataframe Name
    * col : Columns Name    
    '''
    if column is None :
        df = df.describe()
    else:
        df = df[column].describe()
        
    return df


#-----------------------------------------------#
#             Seperating Data Types             #
#-----------------------------------------------#
# function 1
def check_dtypes_1(df):
    '''
    Parameters :
    ------------
    df : Dataframe name 

    Step :
    ------
    > 1. Do iteration for each feature to define which one categorical and nummerical feature. 
    > 2. Columns in dataframe will be seperated based on the dtypes
    > 3. All of the column will be entered to the list that have been created

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

#function 2
def check_dtypes_2(df):
    '''
    Parameters :
    ------------
    df : Dataframe name 
    
    Step :
    > 1. Do iteration for each feature to define which one categorical and nummerical feature. 
    > 2. Columns in dataframe will be seperated based on the dtypes
    > 3. All of the column will be entered to the list that have been created

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

#-----------------------------------------------#
#            Make a List Data Types             #
#-----------------------------------------------#
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

#-----------------------------------------------#
#              Count Unique Column              #
#-----------------------------------------------#
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

#-----------------------------------------------#
#            Checking Missing Value             #
#-----------------------------------------------#
def check_missing_value(df):
    '''
    The function is used for checking missing value of each feature. 

    Parameters :
    ------------
    df : Dataframe Name
    '''
    #count the number of missing value 
    total = df.isnull().sum()
    percent = round(df.isnull().sum()/len(df)*100,2)
    missing  = pd.concat([total, percent], axis=1, keys=['Total_Missing', 'Percent(%)'])
    missing.sort_values(by='Total_Missing', ascending=False, inplace=True)
    
    return print(missing.head(20))


#-----------------------------------------------#
#               Drop Missing Value              #
#-----------------------------------------------#
def Drop_Missing_value(df, threshold = None):
    '''
    Parameters :
    ------------
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
    size_df   = df.shape[0]

    # Define Column list that will we removed
    dropcol   = []

    #looping to take the number of null of every feature 
    for col in df.columns :
        if (df[col].isnull().sum()/size_df >= threshold):
            dropcol.append(col)
    df = df.drop(dropcol, axis =1)

    return df

#-----------------------------------------------#
#                 Normality test                #
#-----------------------------------------------#
def normality_test(df, methods = None , numerical_columns = None, alpha = None):
    '''
    Normality test function is used for checking normality of each numerical features.
    The methods that used consist two methods which are univariate and multivariate.
    Univariate method is representing univariate feature and show normality graph, it is qqplot.
    Multivariate methode is representing multivariate features that will be saw the normality.   
    
    parameters :
    ------------

    df          :  object, dataframe 
            the dataset that will be used 
    methods     :  string, default(univariate)
            > univariate   : including one feature and shows the qqplot 
            > multivariate : including several of features and it will be created into table
    alpha       :  float, default(0.05)
            alpha is a threshold for determining when the hypotesis is reject or accepted 
    
    '''
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
            p    = round(p,3)
            result.append(p)
        
        result = np.array(result)
        column = np.array(columns)
        table  = pd.DataFrame({'feature':column, 'p_value':result})
        table['alpha'] = 0.05
        table['decision'] = np.where(table['p_value'] >= 0.05,'normal distribution', 'not normally distributed')
                    
        return table 

#-----------------------------------------------#
#          Univariate outlier detection         #
#-----------------------------------------------#
def univariate_detect_outliers(df,col = None):
    '''
    parameters : 
    ------------
    df   : object, dataframe
    col  : string 
    
    '''

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

#-----------------------------------------------#
#           muliple outlier detection           #
#-----------------------------------------------#
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
#                     DATA CLEANING                     #
#########################################################

#-----------------------------------------------#
#            Droping Multiple Columns           #
#-----------------------------------------------#
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

#-----------------------------------------------#
#               Change Data Type                #
#-----------------------------------------------#
def change_dtypes(df, col_int, col_float): 
    '''
    parameter :
    -----------
    df           :   object, Dataframe
                  Dataframe that used
    col_int      :   string 
                  integer column from the dataframe
    col_float    :   string
                  float column from the dataframe
 
    '''
    df[col_int] = df[col_int].astype('int32')
    df[col_float] = df[col_float].astype('float32')

#-----------------------------------------------#
#               Convert Cat to Num              #
#-----------------------------------------------#
def convert_cat2num(df):
    '''
    parameter : 
    -----------
    df              :  object, Dataframe
                     The dataframe that used    
    '''
    # Convert categorical variable to numerical variable
    num_encode = {'col_1' : {'YES':1, 'NO':0},
                  'col_2'  : {'WON':1, 'LOSE':0, 'DRAW':0}}  
    df.replace(num_encode, inplace=True)

#-----------------------------------------------#
#          Removing column white space          #
#-----------------------------------------------#
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

#-----------------------------------------------#
#         Concat 2 Columns with Strings         #
#-----------------------------------------------#
#Concatenate two columns with strings (with condition)
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

#Concatenate two columns with strings (with condition)
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

#########################################################
#                  FEATURE ENGINEERING                  #
#########################################################
#-----------------------------------------------#
#             Filling Missing Value             #
#-----------------------------------------------#
def fill_missing(df, feature_list = None , vartype = None ):
    '''
    Documentation :
    ---------------
    df              : object, dataframe
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
            df[col] = df[col].fillna(df[col].median())
    
    # filling categorical data with modus  
    if vartype == 'categorical' :
      for col in feature_list:
        df[col] = df[col].fillna(df[col].mode().iloc[0])



#########################################################
#                     VISUALIZATION                     #
#########################################################
#-----------------------------------------------#
#               Univariate Analysis             #
#-----------------------------------------------#
def univariate_plot(df, column, vartype = None, hue = None ):
    '''
    Parameters :
    -----------
    Univariate function will plot the graphs based on the parameters.
    df      : dataframe name
    column  : Column name
    vartype : variable type : continuos or categorical
                (0) Continuos/Numerical   : Distribution, Violin & Boxplot will be plotted.
                (1) Categorical           : Countplot will be plotted.
    hue     : It's only applicable for categorical analysis.
    '''
    if vartype is None :
        vartype = 0

    if vartype == 0:
        fig, ax=plt.subplots(nrows = 3, ncols=1,figsize=(12,12))
        # Distribution Plot
        ax[0].set_title("Distribution Plot",fontsize = 10)
        sns.distplot(df[column], kde=False, fit=stats.gamma, color='darkblue', label = column, ax=ax[0])
        
        # Violinplot 
        ax[1].set_title("Violin Plot",fontsize = 10)
        sns.violinplot(data= df, x=column, color = 'limegreen', inner="quartile", orient='h', ax=ax[1])
        
        #Boxplot
        ax[2].set_title("Box Plot",fontsize = 10)
        sns.boxplot(data =df, x=column ,color='cyan',orient="h",ax=ax[2])
        
        fig.tight_layout()
        
    if vartype == 1 :
        #Count plot 
        fig = plt.figure(figsize=(12,6))
        plt.title('Count Plot',fontsize = 20)
        ax=sns.countplot(data=df, x=column, palette="Blues_r")
        ax.set_xlabel(column, fontsize = 15)
        ax.tick_params(labelsize=12)

#-----------------------------------------------#
#               Bivariate Analysis              #
#-----------------------------------------------#
def bivariate_plot(df, xcol, ycol, plot_type, hue = None, title= None):
    '''
    Parameters :
    ------------
    Bivariate function will plot the graphs based on the parameters.
    df        : dataframe name
    xcol      : X Column name
    ycol      : Y column name
    plot_type : plot type : scatter plot, boxplot, and violin plot 
                  (0) Scactter plot     : graph between xcol(numerical) and ycol(numerical) 
                  (1) Boxplot           : graph between xcol(categorical) and ycol(numerical)
                  (2) Violin plot       : graph between xcol(categorical) and ycol(numerical)
    hue       : name of variables in ``data`` or vector data, optional Grouping variable that 
                  will produce points with different colors. (String dtype)

    title     : String, default 'Bivariate Plot'
    			  title of the graph
    '''
    if title == None :
        title = 'Bivariate Plot'
        
    # Scatter plot 
    if plot_type == 0 :
        plt.figure(figsize=(12,8))
        ax = sns.scatterplot(data=df, x=xcol, y=ycol, s=150)
        #title of graph
        ax.axes.set_title(title,fontsize = 20 )
        ax.set_xlabel(xcol, fontsize = 15)
        ax.set_ylabel(ycol, fontsize = 15)
        ax.tick_params(labelsize=12)
        
    #boxplot
    if plot_type == 1 : 
        plt.figure(figsize = (12, 7))
        ax =sns.boxplot(data=df, x=xcol, y=ycol, hue = hue)
        plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
        plt.xticks(rotation=45)
        #title of graph
        ax.axes.set_title(title,fontsize = 20 )
        ax.set_xlabel(xcol, fontsize = 15)
        ax.set_ylabel(ycol, fontsize = 15)
        ax.tick_params(labelsize=12)
        
    #violinplot 
    if plot_type == 2 :
        plt.figure(figsize = (12, 7))
        ax = sns.violinplot(data=df, x=xcol, y=ycol,  hue = hue)
        plt.xticks(rotation=45)
        #title of graph
        ax.axes.set_title(title,fontsize = 20 )
        ax.set_xlabel(xcol, fontsize = 15)
        ax.set_ylabel(ycol, fontsize = 15)
        ax.tick_params(labelsize=12)

#-----------------------------------------------#
#             Multivariate Analysis             #
#-----------------------------------------------#
def multivariate(df, column, plot_type ):
    '''
    Parameters :
    ------------
    Multivvariate function will plot the graphs based on the parameters.
    df      : dataframe name 
    column  : Column name (array)
    plot : plot_type : hitmap and pairplot
                (0) Hitmap    : Hitmap graph will be plotted.
                (1) pairplot  : pairplot graph will be plotted.
    '''
    # hitmap plot
    if plot_type == 0 :
        corrMatt = df[column].corr()
        mask = np.array(corrMatt)
        mask[np.tril_indices_from(mask)] = False
        fig = plt.subplots(figsize=(12,10))
        #fig.set_size_inches(20,5)
        sns.heatmap(corrMatt, mask=mask,vmax=0.9, square=True,annot=True, fmt = ".2f")
        
    # pairplot 
    if plot_type == 1 :
        pairplot = sns.pairplot(df[column], size=2, aspect=2,
                                plot_kws=dict(edgecolor="k", linewidth=0.5),
                                diag_kind="kde", diag_kws=dict(shade=True))
        fig = pairplot.fig 
        fig.subplots_adjust(top=0.90, wspace=0.2)
        fig.suptitle('Pairplot', fontsize=15)

#-----------------------------------------------#
#                  Density Plot                 #
#-----------------------------------------------#
def multi_density_plot (df, columns, groupBy=None, plot_per_row=None):   
    '''
    Parameters :
    ------------
    df           : DataFrame name
    columns      : array of numeric columns' name
    groupBy      : string
    plot_per_row : integer
    '''
    if plot_per_row is None:
        plot_per_row = 2
    
    # Counter and plot number
    n = 1
    # Total plots
    size = len(columns)
    # Total rows
    total_row = round(size/plot_per_row + 1, 0)
    
    if groupBy is not None:
        plt.figure(figsize=(20, total_row * 5))
        for column in columns:
            plt.subplot(total_row, plot_per_row, n) # (row, column, panel number)
            df.groupby(groupBy)[column].plot.density(title=column)
            plt.legend()
            n = n + 1
        plt.show()
    else:
        plt.figure(figsize=(20, total_row * 5))
        for column in columns:
            plt.subplot(total_row, plot_per_row, n) # (row, column, panel number)
            df[column].plot.density(title=column)
            plt.legend()
            n = n + 1
        plt.show()

#-----------------------------------------------#
#           TimeSeries Visualization            #
#-----------------------------------------------# 
def timeseries_plot(df, feature, title = None):
    '''
    Parameters :
    ------------
    Multivariate function will plot the graphs based on the parameters.
    df      : dataframe name
    column  : Column name 
    title   : String, Default Time Series Graph 
    '''
    #default title
    if title is None:
        title = 'Time Series Graph'
    
    plt.figure(figsize=(15, 7))
    plt.plot(df[feature], color= 'blue')
    plt.title(title, fontsize= 15)
    plt.grid(True)
    plt.legend(title="actual", loc=4, fontsize='small', fancybox=True)

#-----------------------------------------------#
#       Strong Correlation Visualization        #
#-----------------------------------------------#
#strong correlation 
def scatter_corr(df, feature_Strong_corr = None, ycol = None):
    strong_corr_list = feature_Strong_corr 
    strong_corr = df[strong_corr_list]
    
    features_to_analyse = [col for col in strong_corr_list if col in strong_corr]
    features_to_analyse.append(ycol)
    features_to_analyse
    
    #Regplot
    fig, ax = plt.subplots(round(len(features_to_analyse)/3), 3, figsize = (15, 10))
    for i, ax in enumerate(fig.axes):
        if i < len(features_to_analyse)-1:
            sns.regplot(x=features_to_analyse[i],y=ycol, data=df[features_to_analyse], ax=ax)
    fig.tight_layout()


#########################################################
#                   FEATURE SELECTION                   #
#########################################################
#--------------------------------------------------------#
#         Define Strong Correlation from Dataset         #
#--------------------------------------------------------# 
def strong_corr(df, ycol=None):
    '''
    Documentation :
    --------------
    * Df   : Dataset name
    * ycol : The target column of the dataframe
    '''
    df_num_corr = df.corr()[ycol][:-1] # -1 because the latest row is SalePrice
    strong_corr = df_num_corr[abs(df_num_corr) > 0.6].sort_values(ascending=False)
    corr = pd.Series(strong_corr, name='Strong Correlation')
    table = pd.concat([corr], axis = 1) 
    print ('Columns have strong correlation :')
    print(table)
    list_features = table.index.values 
    
    return list_features

#--------------------------------------------------------#
#       Correlation between catagotrical features        #
#--------------------------------------------------------# 
#Module Chi-square
def ChiSquare_Test(df, ycol, xcol, alpha = None):
    '''
    Parameters : 
    --------------
    df     : object, dataframe
        Dataframe that will be used 
    ycol   : string
        target feature that will be used 
    xcol   : list, srting
        a few of the features that will be used  
    alpha  : float, default(0.05)
        
    
    ex:
    # 1. Define Dataframe
    df_categ = df_categ_list
    
    # 2. Input y column  
    y = 'SaleType'
    
    # 3. Input X columns
    x =['Street','LandContour','LandSlope']
   
   # 4. Significance Level 
    a = 0.05
    
    # 5. Chi-Square Analisys 
    ChiSquare_Test(df = df_categ_list, ycol=y, xcol= x, alpha= a)
    '''

    if alpha == None :
        alpha = 0.05

    result = {}
    #Looping for every x variable in Dataframe
    for x in xcol :
        crosstab = pd.crosstab(df[ycol],df[x])
        _, p, _, _ = chi2_contingency(crosstab)
        
        #logic
        if p <= alpha :
            result[x] = 'Correlated'
        else :
            result[x] = 'Not Correlated'
    
    #The result will be created to the pandas dataframe 
    df = pd.DataFrame.from_dict(result, orient='Index', columns=['Decision'])
    df.index.name='Colname'
    
    return df

#--------------------------------------------------------#
#      Features selection using Logistic Regression      #
#--------------------------------------------------------# 

def log_regression(df, ycol = None, alpha = None):
    '''
    logistic regression can be used for feature selection before modeling.
    the ways is taking p_value from each feature that produced from summary modeling. 

    parameters :
    ------------
    df      :  object, dataframe
        dataframe that will be used 
    ycol    :  string, target column 
        target column that had owned by dataframe 
    alpha   : float, default(0.05)
        alpha is a threshold for determining when the hypotesis is reject or accepted
    
    '''
    if alpha == None :
        alpha = 0.05
    #determine y and x variable     
    y = df[ycol]
    x = df.drop([ycol], axis = 1)
    #regression logistic
    model = sm.Logit(y,x)
    #regression 
    result = model.fit()
    #get p-values
    result             = round(result.pvalues,3)
    result             = pd.DataFrame(result, columns=['p_value'])
    result.index       = result.index.set_names(['features'])
    #result            = result.reset_index().rename(columns = {result.index.name : 'features'})
    result['alpha']    = alpha
    result['decision'] = np.where(result['p_value'] <= alpha, 'Correlated', 'Not Correlated')
    #result             = result['decision']
    
    return result 

#--------------------------------------------------------#
#       Features selection using Linear Regression       #
#--------------------------------------------------------# 

def lin_regression(df, ycol = None, alpha=None):
    '''
    linear regression can be used for feature selection before modeling.
    the ways is taking p_value from each feature that produced from summary modeling. 

    parameters :
    ------------
    df      :  object, dataframe
        dataframe that will be used 
    ycol    :  string, target column 
        target column that had owned by dataframe 
    alpha   : float, default(0.05)
        alpha is a threshold for determining when the hypotesis is reject or accepted
    
    '''
    if alpha == None :
        alpha = 0.05 
    #determine y and x variable 
    y = df[ycol]
    x = df.drop([ycol], axis = 1)
    #build model 
    model = sm.OLS(y,x)
    #fit model OLS 
    result = model.fit()
    #get p-values
    result             = round(result.pvalues,3)
    result             = pd.DataFrame(result, columns=['p_value'])
    result.index       = result.index.set_names(['features'])
    #result            = result.reset_index().rename(columns = {result.index.name : 'features'})
    result['alpha']    = alpha
    result['decision'] = np.where(result['p_value'] <= alpha, 'Correlated', 'Not Correlated')
    #result             = result['decision']
    return result 