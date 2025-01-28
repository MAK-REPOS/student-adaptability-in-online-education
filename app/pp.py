import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


def data() :
    data = pd.read_csv(r'students_adaptability_level_online_education.csv')
    sub_plot_data = pd.read_csv(r'students_adaptability_level_online_education.csv')

    return data,sub_plot_data

def preprocessing(data,flag) :
    cat_cols = []
    for i in data.columns:
        if np.dtype(data[i]) == 'object' :
            if i == 'Adaptivity Level' :
                continue
            else :
                cat_cols.append(i)

    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    for i in cat_cols :
        le = LabelEncoder()
        data[i] = le.fit_transform(data[i])
        encoders[i] = le
    

    if flag == 'Full prediction' :
        x = data.drop('Adaptivity Level',axis = 1)
        y = data['Adaptivity Level']

        return x,y,cat_cols,encoders,data
    return cat_cols,encoders,data

def train(x,y) :
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state = 42)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics

    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)

    return x_test,y_test,clf
    
def prediction(clf,x_test) :
    y_pred = clf.predict(x_test)

    return y_pred

def accuracy(y_test,y_pred) :
    acc = accuracy_score(y_test,y_pred)

    return acc


def pie_chart(y_pred) :

    dict = {i : 0 for i in y_pred}
    for i in y_pred :
        dict[i] += 1
    s_dict = pd.Series(dict)
    plt.switch_backend('Agg')

    plt.figure(figsize = (10,8))
    plt.pie(s_dict,labels = s_dict.index,autopct = '%1.1f%%',startangle = 140)
    plt.title('Proportion of adaptability')

    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png')
    buffer.seek(0)
    plt.close()

    pie_data = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return pie_data

def bargraphs(sub_plot_data) :

    nrows = 7
    ncols = 2

    
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 25))
    axes = axes.flatten()

    for i, k in enumerate(sub_plot_data.columns[:-1]): 
        sns.countplot(ax=axes[i], x=k, hue='Adaptivity Level', data=sub_plot_data, palette='pastel')
        axes[i].set_title('Adaptivity Level by ' + k)
        axes[i].set_xlabel(k)
        axes[i].set_ylabel('Count')

    for i in range(len(sub_plot_data.columns) - 1, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer,format = 'png')
    buffer.seek(0)

    plt.close()

    bar_data = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    #plt.show()

    return bar_data
    