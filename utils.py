import numpy as np
import pandas as pd
import scipy.stats as st
from plotnine import *
import seaborn as sns
from sklearn.linear_model import LinearRegression

#cleaning
def check_missing(df):
    per_missing = df.isnull().mean()
    missing_df = pd.DataFrame({'col': df.columns, 'per_missing': per_missing})
    missing_df = missing_df.sort_values('per_missing',ascending=False).reset_index(drop=True)
    missing_df['col'] = pd.Categorical(missing_df.col, categories=missing_df.col, ordered=True)
    return missing_df

def missingify(df, impute_fn=np.nanmedian, impute_token='xxna'):
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_cat = df.select_dtypes(exclude=[np.number]).copy()
    #numerical
    for col in df_num.columns:
        df_num[f'{col}_missing'] = df_num[col].isna().astype('float')
        df_num.loc[df_num[col].isna(),col] = impute_fn(df_num[col])

    #categorical
    print(df_cat.columns)
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].fillna(impute_token)
    return pd.concat([df_cat,df_num],1).reset_index(drop=True)

def check_mode(df):
    mode_df = []
    for col in df.columns:
        x = df[col].value_counts()
        mode_df.append({'col':col, 'value':x.index[0], 'per_mode': list(x)[0]/df.shape[0],
                       'nb_value':len(x)})
    mode_df = pd.DataFrame(mode_df)[['col','value','per_mode','nb_value']]\
        .sort_values('per_mode',ascending=False)
    mode_df['col'] = pd.Categorical(mode_df.col, categories=mode_df.col, ordered=True)
    return mode_df.reset_index(drop=True)

def value_dist(df,col):
    x = pd.DataFrame(df[col].value_counts()).reset_index()
    x.columns = ['value','cnt']
    x['per'] = x.cnt / x.cnt.sum()
    return x

def fuzzy_dict(df, col, fuzz_fn, th=90):
    values = df[col].unique()
    result = {}
    for i in values:
        for j in values:
            if fuzz_fn(i,j) > th:
                result[i] = j
    return result

def other_dict(df, col, th=0.03):
    value_df = value_dist(df,col)
    other_values = list(value_df[value_df.per<th].value)
    result = {}
    for val in value_df.value: 
        if val in other_values:
            result[val] = 'others'
        else:
            result[val] = val
    return result

def otherify(df, col, d):
    df_n = df.copy()
    df_n[col] = df_n[col].map(lambda x: d[x])
    return df_n

def remove_outliers(df,col):
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3-q1
    df = df[(df[col] < q3+1.5*iqr)&(df[col] > q1-1.5*iqr)]
    return df.reset_index(drop=True)

#visualization
def value_cutoff_plot(df,col):
    value_df = value_dist(df,col)
    g = (ggplot(value_df.reset_index(),aes(x='index',y='per')) + geom_point(size=0.5) + 
         geom_line() + theme_minimal())
    return g

def cat_plot(df,col):
    g = (ggplot(df,aes(x=col,fill=col)) + 
         geom_bar(stat='bin', #histogram
                  binwidth=0.5, #histogram binwidth
                  bins=len(df[col].unique())) + #how many bins
         theme(axis_text_x=element_blank())
        )
    return g

def numdist_plot(df, num,cat, geom=geom_density(alpha=0.5), no_outliers=True):
    if no_outliers:
        new_df = remove_outliers(df,num)
    else:
        new_df = df.copy()
    g = (ggplot(new_df,aes(x=num, fill=cat)) +
          geom 
        )
    return g

def numcat_plot(df,num,cat, no_outliers=True, geom=geom_boxplot()):
    if no_outliers:
        new_df = remove_outliers(df,num)
    else:
        new_df = df.copy()
    g = (ggplot(new_df, aes(x=cat,y=num)) +
         geom 
        )
    return g

def catcat_plot(df, cat_dep, cat_ind):
    g = (ggplot(df,aes(x=cat_dep,fill=cat_dep)) + geom_bar() + 
         theme(axis_text_x = element_blank()) +
         facet_wrap(f'~{cat_ind}',scales='free_y')) + theme(panel_spacing_x=0.5)
    return g

def value_dist_plot(df,bins=30):
    num_m = df.melt()
    g = (ggplot(num_m,aes(x='value')) +
         geom_bar(stat='bin', bins=bins) +
         facet_wrap('~variable', scales='free') + #facetting by variable
         theme_minimal() + theme(panel_spacing_x=0.5)
        )
    return g

def jointplot(df,col_x, col_y, no_outliers=True, kind='reg'): #'scatter','resid','reg','hex','kde','point'
    if no_outliers:
        new_df = remove_outliers(df,col_x)
        new_df = remove_outliers(new_df,col_y)
    else:
        new_df = df.copy()
    return sns.jointplot(new_df[col_x],new_df[col_y],kind=kind)
    
def qq_plot(df,col):
    qq, reg = calc_qq(df,col)
    g = (ggplot(qq,aes(x='theoretical_q',y='sample_q')) + 
        geom_point() + #plot points
        geom_abline(slope=1,intercept=0,color='red') + #perfectly normal line
        stat_function(fun=lambda x: x*reg.coef_[0][0]) + #linear estimation
        ggtitle(f'y= {np.round(reg.coef_[0][0],2)} * x')+ #display equation
        labs(x='Theoretical Quantiles (normalized)', y='Sample Qunatiles (normalized)'))
    return g

def boxcox_plot(df, col, ls = [i/10 for i in range(-30,31,5)]):
    lamb_df = boxcox_lamb_df(df[col],ls)
    g = (ggplot(lamb_df, aes(x='lamb',y='coef',group=1)) + 
         geom_point() + geom_line())
    return g

#transformation
def calc_qq(df,col):
    sample_qs = [(np.percentile(df[col],i)-np.mean(df[col]))/np.std(df[col]) for i in range(5,100,5)]
    theoretical_qs = [st.norm.ppf(i/100) for i in range(5,100,5)]
    qq = pd.DataFrame({'sample_q':sample_qs,'theoretical_q':theoretical_qs})
    reg = LinearRegression(fit_intercept=False).fit(np.array(qq['theoretical_q'])[:,None], 
                                 np.array(qq['sample_q'])[:,None])
    return qq, reg

def boxcox(ser,lamb=0):
    ser+= 1 - ser.min()
    if lamb==0: 
        return np.log(ser)
    else:
        return (ser**lamb - 1)/lamb
    
def boxcox_lamb_df(ser, ls = [i/10 for i in range(-30,31,5)]):
    coefs = []
    for l in ls:
        df = pd.DataFrame.from_dict({'val': boxcox(ser,l)})
        qq, reg = calc_qq(df,'val')
        coefs.append(reg.coef_.squeeze().item())
    return pd.DataFrame({'lamb':ls,'coef':coefs})

def boxcox_lamb(ser, ls = [i/10 for i in range(-30,31,5)]):
    df = boxcox_lamb_df(ser,ls)
    return df.lamb[df.coef.idxmax()]