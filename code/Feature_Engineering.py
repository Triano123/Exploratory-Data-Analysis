#import module analytics 
import pandas as pd 
import numpy as np 

#########################################################
#             FILLING MISSING VALUE FUNCTION            #
#########################################################
# Filling Missing value 
def fill_missing(df, feature_list = None , vartype = None ):
    '''
    Documentation :
    ---------------
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
            df[col] = df[col].fillna(df[col].median())
    
    # filling categorical data with modus  
    if vartype == 'categorical' :
      for col in feature_list:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

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