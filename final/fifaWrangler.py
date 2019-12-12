# fifaWrangler.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def vis_null(df):
    sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")
    plt.show()

def wrangle(df, plots=True, print_step=False):
    '''
    Data cleaning the object dataframe to only include columns needed for ml
    '''
    print('\n-- WRANGLING DATA --')
    keepList = ['Position','Age','Overall','Preferred Foot','Body Type','Height',
                'Weight', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
                'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',
                'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
                'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
                'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
                'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']
    df = df.loc[:,keepList] # removing unnecessary columns
    if plots:
        vis_null(df)
    df = df.dropna(how='any', axis=0) # removing null values (b/c need position label)
    if plots:
        vis_null(df)
    if print_step:
        print(df.dtypes) # checking data types
        print(df['Body Type'].value_counts())
        print(df['Preferred Foot'].value_counts())
    df['Body Type'] = df['Body Type'].map({'Normal':0.0,'Lean':1.0,'Stocky':2.0}).fillna(0.0) # changing str values to floats and filling na with normal
    df['Preferred Foot'] = df['Preferred Foot'].map({'Right':0.0,'Left':1.0})
    df['Height'] = [((12*float(str(x).split("'")[0])) + float(str(x).split("'")[1])) for x in df.Height]
    df['Weight'] = df['Weight'].str[:3].astype('float')
    df['Age'] = df['Age'].astype('float')
    df['Overall'] = df['Overall'].astype('float')
    if print_step:
        print(df['Body Type'].value_counts())
        print(df['Preferred Foot'].value_counts())
        print(df['Height'].value_counts())
        print(df['Weight'].value_counts())
        print(df.dtypes)
        print(df.Position.value_counts())
    print('Final datatypes: \n\n', df.dtypes)
    #df = df[df['Overall'] >= 75.0]
    # removing positions with less than 100 observations to train on
    df = df[df.Position.isin(list(df.Position.value_counts()[:10].index))]
    print(df.Position.value_counts())
    return df

if __name__=='__main__':
    df = pd.read_csv('fifa19.csv')
    df = wrangle(df, plots=True, print_step=False)
    df = df.to_csv('fifa19wr.csv',index=True)
    print(df.head(20))
