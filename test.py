# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:29:08 2020

@author: Ugur
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sb

figDir = "Figures/"     


df = pd.read_json('syntetic-nd.json', lines = True)    
df = df.set_index("timestamp")
#%% Libraries
from sklearn import preprocessing

def split(): print("\n____________________________________________________________________________________\n")

#Tüm featureler için korelasyon matrisi
def plotCorrelationMatrix(df, graphWidth,save=False):  
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for xd', fontsize=40)
    plt.show()
#Boxplot, gruplama ve korelasyonların hepsini analiz eden all-in-one fonksiyon
    if(save):
        plt.savefig(figDir+'CorrelationMatrix.png')
def intro(df,graph=True,splitPlots=True,EraseNullColumns=False,printCorrelations=True,corrThreshold=0.5,save=False):
    
    dataframe=df.copy()
    
    if(EraseNullColumns==True):  dataframe.dropna(axis=1,inplace=True)

    split()
    print(df)
    split()
    print(dataframe.head(5))
    split()
    
    print(dataframe.info())
    split()
    
    print(dataframe.describe())
    split()
    
#-------------------------------BOXPLOTFEATURES-----------------------------      
    
    
    if(graph):

        if(splitPlots==True):
            print("                         ___BOXPLOTFETURES")

            for column in dataframe.columns:
                if(dataframe[column].dtype==np.int or dataframe[column].dtype==np.float):
                    plt.figure()
                    dataframe.boxplot([column])
                    if(save):
                        plt.savefig(figDir+'{}.png'.format(column))
                    
        else:
            dataframe.boxplot()
            
    #If unique values of columns is under 10, print unique values with considered column


#-------------------------------GROUPBY-----------------------------        

    print("                         _____GROUPBY____")

    for column in dataframe.columns:    
        unique_values=dataframe[column].unique()
        if(unique_values.size<=10):
            print(column,": ",unique_values)
            print("\nGrouped By: ",column,"\n\n",dataframe.groupby(column).mean())
            split()
            print("\n")
            
        
#-------------------------------CORRELATIONS-----------------------------        
    if(printCorrelations==True):
        print("                         ____CORRELATIONS____")
        corrByValues= dataframe.corr().copy()
        flag = False
        corr_matrix=abs(corrByValues>=corrThreshold)
        columns= corr_matrix.columns
        for i in range(columns.size):
            for j in range(i,columns.size):
                iIndex=columns[i]
                jIndex=columns[j] 
                if (i!=j and corr_matrix[iIndex][jIndex]==True and (len(df[iIndex].unique())!=1 and len(df[jIndex].unique())!=1 )):
                    sign = "Positive"
                    if(corrByValues[iIndex][jIndex]<0): sign="Negative"
                    split()
                    flag = True
                    print(iIndex.upper(), " has a " ,sign," correlation with ",jIndex.upper(),": {} \n".format(corrByValues[iIndex][jIndex]))
        
        plt.show()
        plotCorrelationMatrix(df,30)       
        
        split()
        if(not flag):
            print("No Correlation Found") 
    return dataframe

#KDE dağılımı ile featureları plotlar
def plotCols(df,time,save=False):
    
    for col in df.columns:
        if(df[col].dtype==np.int or df[col].dtype==np.float):
            if(len(df[col].unique())>1):
                fig = df.plot(x=time,y=col,kind="kde", title = "{}-{} KDE".format(time,col))   
                if(save):
                    fig.get_figure().savefig(figDir+"{}-kde.png".format(time+"-"+col))
                plt.show()
            plt.plot(df[time],df[col]) 
            plt.title("{}-{}".format(time,col))
            plt.show()
            if(save):
                plt.savefig(figDir+'{}.png'.format(time+"-"+col))
        
#Verilen feature'ları scatter ile Y'ye göre karşılaştırır.        
def XCorrWithY(df, X, Y):
    for col in  X:
        print(col,"-",Y)
        plt.scatter(df[col],df[Y]) 
        plt.title("{}-{}".format(col,Y))
        plt.show()    
#Dataframeyi normalize eder. (Preprocessing)        
def normalizedf(df,offset=0):
    min_max_scaler = preprocessing.MinMaxScaler() 
    new = df.copy()
    cols = df.columns[offset:]
    new[cols] = (min_max_scaler.fit_transform(new[cols]) )  
    return new    
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset.iloc[i:(i+look_back), :]
		dataX.append(a.to_numpy())
		dataY.append(dataset.iloc[i + look_back, -1]) 
	return np.array(dataX), np.array(dataY)  


#%% Preprocessing

df = normalizedf(df)

X_f, Y_f = create_dataset(df.drop(columns=["Feature2"]),17)
#%% Visualize and Analyze dataframe
intro(df)
df.reset_index().plot(x="timestamp",y="Feature1" )
df.reset_index().plot(x="timestamp",y="Feature2" )

#Bir senörün anomlisi, diğer sensörden bağımsız. Anomalileri arasında ilişkileri yok.
#2 Sensör arasında Korelasyonun 0.54 değerinde olması, frekanslarının aynı olmasından kaynaklanıyor olmalı
#%% Sklearn Anomaly Detection Libraries
from sklearn.svm import OneClassSVM #Bu data üzerinde Başarılı Değil
from sklearn.ensemble import IsolationForest  #Bu data üzerinde başarılı değil
from sklearn.neighbors import LocalOutlierFactor 
from kenchi.outlier_detection.statistical import HBOS

modelsNotFitted = [ LocalOutlierFactor(leaf_size=10 , novelty=False) ] 
X = X_f.reshape(X_f.shape[:-1]) 
#Anomali olup olmadığına bak
for model in modelsNotFitted: 
    model.fit_predict(X) 
    lof = model.negative_outlier_factor_
    plt.plot(lof/min(lof),color="orange") 
    plt.title("LocalOutlierFactor On Feature 1")
    plt.plot(Y_f)
    plt.show()      
#LocalOutlierFactor başarılı    

#%% LUMINOL    
from luminol import  anomaly_detector 
df = df.reset_index()
df["timestamp"] = (df["timestamp"].astype('uint64') / 1e6).astype('uint32')
df.set_index("timestamp")
a = df.drop(columns=["Feature2"]).to_dict("series")["Feature1"].to_dict()
detector = anomaly_detector.AnomalyDetector(a,algorithm_name="exp_avg_detector",score_threshold = 0.2 ) 
anomalies = detector.get_anomalies() 
 
time_periods = [] 
values = []
for key in (a): 
    time_periods.append(key)
    values.append(a[key])
plt.plot(time_periods,values)     
       
anom_times = []
anom_values = []
for anom in anomalies :  
    anom_times.append(np.arange(anom.start_timestamp,anom.end_timestamp))
    anom_values.append(anom.anomaly_score)
    s, e = anom.start_timestamp, anom.end_timestamp
    v = []
    for i in range(s,e  ):
        v.append(a[s])
    timerange =  np.arange(s,e)   
    plt.plot(timerange,v, c ="r",marker= 'o') 
plt.title("Luminol on Feature 1")
plt.show( )    

#%% Robust Random Cut Forest Algorithm

import rrcf
X = df["Feature1"].tolist()
 
TREE_COUNT = 100
SHINGLE_COUNT = 8
TREE_SIZE = 200
forest = [] 
for i in range(TREE_COUNT):
    tree = rrcf.RCTree()
    forest.append(tree)
points = rrcf.shingle(X, size=SHINGLE_COUNT)

avg_codisp = {}

 
for index, point in enumerate(points): 
    for tree in forest: 
        if len(tree.leaves) > TREE_SIZE:
            tree.forget_point(index - TREE_SIZE) 
        tree.insert_point(point, index=index) 
        if not index in avg_codisp:
            avg_codisp[index] = 0
        avg_codisp[index] += tree.codisp(index) / TREE_COUNT
        
time_periods = [] 
values = []
for key in avg_codisp:
    time_periods.append(key)
    values.append(avg_codisp[key])

PLOTSCALE= 5    
values = np.array(values)/(PLOTSCALE* max(values)) 
#%%
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
plt.title("Robust Random Cut Forest Algorithm on Feature 1")

sb.lineplot(time_periods,values,label="Anomaly Scores"  )
sb.lineplot(time_periods,X[SHINGLE_COUNT-1:],label="Values")  
#%% AUTOENCODER NOVELTY
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM, Dropout, LeakyReLU  ,RepeatVector,TimeDistributed

from keras.optimizers import Adam, SGD, Adamax,RMSprop  
import tensorflow as tf 
from keras.callbacks import EarlyStopping, ModelCheckpoint
# ilk 4000'de anomaly yok
X_train,_ = create_dataset(df.drop(columns="Feature1")[:4000],200)
model = Sequential()
model.add( LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(Dense(10 ))
model.add( Dropout(rate=0.2))  
model.add(Dense(64))
model.add( Dropout(rate=0.2))
model.add( Dense ( units = X_train.shape[1] ) ) 
          
model.compile(loss='mse', optimizer='rmsprop')
history = model.fit(
    X_train, X_train.reshape( X_train.shape[:-1] ),
    epochs=300,
    batch_size=300,
    validation_split=0.05,
    shuffle=False,
    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min'),
                       ModelCheckpoint("AE1.h5",monitor='val_loss', save_best_only=True, mode='min', verbose=1)]
    )
          
#%% TEST AUTOENCODER
model = load_model("AE1.h5")  
test,_ = create_dataset(df.drop(columns="Feature1")[4000:] ,200)
#%% Plot Original and Reconstructed Data
plt.title("AutoEncoder on Feature 1\nOriginal vs Reconstructed Values")
plt.plot(test.reshape(test.shape[:-1]),c="red")  
plt.plot(model.predict(test),c="orange") 
plt.show()

#%% Plot Training History
sns.lineplot(np.arange(len(history.history["loss"])),history.history["loss"], label="Loss")
sns.lineplot(np.arange(len(history.history["loss"])),history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training")
plt.show()