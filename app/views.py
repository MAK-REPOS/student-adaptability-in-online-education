from django.shortcuts import render
from django.http import JsonResponse
from . import pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def index(request) :

    return render(request,'index.html')

def training(request) :

    return render(request,'training.html')

def file_upload(request) :
    if request.method == 'POST' :
        training_file = request.FILES['file']
        data = pd.read_csv(training_file)
        
        
        return data

def train(request) :

    
    global clf

    data = file_upload(request)
    columns = data.columns[:-1]
    request.session['columns'] = columns.tolist()
    sub_plot_data = data
    sub_data_columns = sub_plot_data.columns

    request.session['sub_data'] = sub_plot_data.values.tolist()
    request.session['sub_columns'] = sub_data_columns.tolist()

    selected_features = list(data.columns)
    flag = 'Full prediction'
    x,y,cat_cols,encoders,data = pp.preprocessing(data,flag)
    x_test,y_test,clf = pp.train(x,y)

    
    y_pred_full = pp.prediction(clf,x_test)

    request.session['y_pred_full'] = y_pred_full.tolist()


    for i in cat_cols :
        le = encoders[i]
        x_test[i] = le.inverse_transform(x_test[i])

    ser = pd.DataFrame(x_test)
    ser['Adaptability'] = y_pred_full
    
    table = ser.to_html(index = False)

    accuracy = pp.accuracy(y_test,y_pred_full)
    accuracy = f'{accuracy * 100:.2f}%' 
    accuracy = np.array(accuracy)
    request.session['accuracy'] = accuracy.tolist()



    x1 = selected_features[0]
    x2 = selected_features[1]
    x3 = selected_features[2]
    x4 = selected_features[3]
    x5 = selected_features[4]
    x6 = selected_features[5]
    x7 = selected_features[6]
    x8 = selected_features[7]
    x9 = selected_features[8]
    x10 = selected_features[9]
    x11 = selected_features[10]
    x12 = selected_features[11]
    x13 = selected_features[12]

    context = {
        'prediction' : table,
        'accuracy' : accuracy,
        'x1' : x1,
        'x2' : x2,
        'x3' : x3,
        'x4' : x4,
        'x5' : x5,
        'x6' : x6,
        'x7' : x7,
        'x8' : x8,
        'x9' : x9,
        'x10' : x10,
        'x11' : x11,
        'x12' : x12,
        'x13' : x13,
        'flag' : flag
    }

    return render(request,'result.html',context)


def manual_prediction(request) :

    flag = 'manual_prediction'
    accuracy = request.session.get('accuracy')
    columns = request.session.get('columns')

    if request.method == 'POST' :
        x11 = request.POST.get('x1')
        x12 = request.POST.get('x2')
        x13 = request.POST.get('x3')
        x14 = request.POST.get('x4')
        x15 = request.POST.get('x5')
        x16 = request.POST.get('x6')
        x17 = request.POST.get('x7')
        x18 = request.POST.get('x8')
        x19 = request.POST.get('x9')
        x120 = request.POST.get('x10')
        x121 = request.POST.get('x11')
        x122 = request.POST.get('x12')
        x123 = request.POST.get('x13')

    input = [x11,x12,x13,x14,x15,x16,x17,x18,x19,x120,x121,x122,x123]
    input = np.array(input)
    input = pd.DataFrame(input)

    cat_cols,encoders,data = pp.preprocessing(input,flag)


    data = pd.DataFrame(data)
    data = data.transpose()
    y_pred = pp.prediction(clf,data)

    for i in y_pred :
        predicted_data = i

    for i in cat_cols :
        le = encoders[i]
        input[i] = le.inverse_transform(input[i])

    
    input = input.transpose()
    input.columns = columns
    table = input.to_html(index = False)

    
    context= {
        'manual_prediction' : predicted_data,
        'table' : table,
        'flag' : flag,
        'accuracy' : accuracy
    }

    return render(request,'result.html',context)

def statistics(request) :

    y_pred_full = request.session.get('y_pred_full')
    sub_plot_data = request.session.get('sub_data')
    sub_columns = request.session.get('sub_columns')
    sub_plot_data = pd.DataFrame(sub_plot_data,columns = sub_columns)

    pie_chart = pp.pie_chart(y_pred_full)
    bar_grapghs = pp.bargraphs(sub_plot_data)

    

    context = {
        'pie_chart' : pie_chart,
        'bar_graphs' : bar_grapghs
    }

    return render(request,'statistics.html',context)