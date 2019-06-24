# import module is nedeed 
from scipy import stats 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style='darkgrid')

#####################################################################
###                    UNIVARIATE ANALYSIS                        ###
#####################################################################
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


#####################################################################
###                    UNIVARIATE ANALYSIS                        ###
#####################################################################
def bivariate_plot(df, xcol, ycol, plot_type, hue = None, title= None):
    '''
    Parameters :
    ------------
    Bivariate function will plot the graphs based on the parameters.
    * df        : dataframe name
    * xcol      : X Column name
    * ycol      : Y column name
    * plot_type : plot type : scatter plot, boxplot, and violin plot 
                  (0) Scactter plot     : graph between xcol(numerical) and ycol(numerical) 
                  (1) Boxplot           : graph between xcol(categorical) and ycol(numerical)
                  (2) Violin plot       : graph between xcol(categorical) and ycol(numerical)
    * hue       : name of variables in ``data`` or vector data, optional Grouping variable that 
                  will produce points with different colors. (String dtype)

    * title     : String, default 'Bivariate Plot'
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

# 3. Multivariate analisys

def multivariate(df, column, plot_type ):
    '''
    Parameters :
    ------------
    Multivvariate function will plot the graphs based on the parameters.
    * df      : dataframe name 
    * column  : Column name (array)
    * plot : plot_type : hitmap and pairplot
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


def multi_density_plot (df, columns, groupBy=None, plot_per_row=None):   
    '''
    Parameters :
    ------------
    * df           : DataFrame name
    * columns      : array of numeric columns' name
    * groupBy      : string
    * plot_per_row : integer
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


#Time Series Graph 
def timeseries_plot(df, feature, title = None):
    '''
    Parameters :
    ------------
    Multivariate function will plot the graphs based on the parameters.
    * df      : dataframe name
    * column  : Column name 
    * title   : String, Default Time Series Graph 
    '''
    #default title
    if title is None:
        title = 'Time Series Graph'
    
    plt.figure(figsize=(15, 7))
    plt.plot(df[feature], color= 'blue')
    plt.title(title, fontsize= 15)
    plt.grid(True)
    plt.legend(title="actual", loc=4, fontsize='small', fancybox=True)
    