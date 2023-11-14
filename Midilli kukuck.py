        # -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:17:12 2023

@author: joxeh
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import math
from math import exp
from math import pi
import warnings
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from lmfit import Model, Parameters

def inicio():
    datos = pd.read_csv("conda.csv")
    print(datos.info()) ##Muestra el tipo de datos
    print(datos.head())## Muestra los primeros 5 datos (encabezado)
    print(datos.describe())##descripcion estadistica de los datos
    
    x= datos.tiempo.values.reshape(-1,1)
    y = datos.mr.values.reshape(-1,1)  ####Redimenciono altiro los valores
    #Para poder utilizrlos en los modelos
    
    #x = np.linspace(0,450,40)#### Crea un arreglo donde los parametros son (punto de partida, punto final, saltos)
    #plt.title("MOISTURE RATIO VS TIME (WITH TIME OF MINUTES")
    #fig,axes = plt.subplots()
    ##axes.scatter(datos["tiempo"],datos["hr"]) Entrega los datos como puntos
    plt.figure(figsize=(9,5))###Se le dalas simenciones al gradico, (ancho, alto)
    #Face color cambia el color de la ventana
    
    plt.plot(x,y,"r") #Cambia el color de la linea
    plt.title("Humedad vs tiempo ")
    plt.ylabel('Contenido de Humedad')
    plt.xlabel('Tiempo  (minutos)')
    plt.legend(["contenido de humedad"],loc =3) #por defecto coloca la leyenda arriba
    #Si no se colocan corchetes no se muestra el mensaje entero
    plt.show()
    print()
    return x,y

    
    
    
def ecuacion_2(x,k,b,n):
    return 0.9835*np.exp(-k*x**n)+b*x



def fraccional():
    datos = pd.read_csv("conda.csv")
    x= (datos.tiempo.values.reshape(-1,1))
    y = datos.mr.values.reshape(-1,1)
    #print(x)
    #print(y)
    model = Model(ecuacion_2,independent_vars=['x'], param_names=["k","b","n"])

    params = Parameters()
    params.add("k", value= 0.000001,min=0.000001,max = 2)
    params.add("b", value=-0.0000001,min = -0.00001,max = -2)
    params.add("n", value= 0.000001,min= 0.000001,max = 2)

    result = model.fit(data=y, params=params, x=x)
    #print(model)

    
    #x = np.linspace(0,22000,410)
    #fig,axes = plt.subplots()
    #axes.scatter(data["tiempo"],data["hr"])
    #axes.plot(x,funcion(x,-1.26811894e-11,1.66027663e+10,-4.09643193e-03,-6.06639361e+09))
    #plt.show()
        
    print(result.fit_report())
    


    
    
    
    
    










def comparacion_2(x_convertida,y):
   

    k = 5.9613e-04
    
    b = -1.6211e-05

    n = 1.15710000

    
   
    #alfa = 1.58453577
    #difusividad2 = 7.2871e-14 t#Su r es mejor pero los datos no ajustan tan bien

    
    y_convertida3 = []
    for elemento in x_convertida:
        
                valor = ecuacion_2(elemento,k,b,n)
                y_convertida3.append(valor)
   
     
  
        
        
  
    
    plt.figure(figsize=(9,5))
    plt.plot(x_convertida,y,"r",x_convertida,y_convertida3,"*") #Cambia el color de la linea
    plt.title("Datos reales vs datos modelados MIDILLI KUCUK")
    plt.ylabel('Contenido de Humedad')
    plt.xlabel('Tiempo (minutos)')
    plt.legend(["contenido de humedad real","contenido de humedad modelado"],loc =1) #por defecto coloca la leyenda arriba
    #Si no se colocan corchetes no se muestra el mensaje entero
    plt.show()
    




def guru():
    datos = pd.read_csv("conda.csv")
    x =datos.tiempo
    y = datos.mr
    
    
    plt.show()
    popt, pcov =   curve_fit(ecuacion_2,x,y,p0= [0.01,0.001,0.6]) 
    kopt,bopt,nopt = popt
    xmodel = np.linspace((min(x), max(x)),1600)
    ymodel = ecuacion_2(xmodel,kopt,bopt,nopt)
    plt.scatter(x,y)
    plt.plot(xmodel,ymodel,color ="red" ) 
    plt.show()
    print()
    print("El señor guru")
    print()
    print(popt)
    
    return




def ecuacion5(x,difusividad2,alfa):
    return (8/9.869604401)*np.exp(-difusividad2*(9.869604401/0.000016)*x**alfa)
    
    
    
    
def delta():
    datos = pd.read_csv("datossegundos.csv")
    x =datos.tiempo
    y = datos.hr
    plt.show()
    popt, pcov =   curve_fit(ecuacion5,x,y,p0= [0.0000000000000001,1]) 
    kopt,bopt = popt
    xmodel = np.linspace((min(x), max(x)),1600)
    ymodel = ecuacion5(xmodel,kopt,bopt)
    plt.scatter(x,y)
    plt.plot(xmodel,ymodel,color ="orange" ) 
    plt.show()
    print()
    print("El señor guru")
    print()
    print("esto son los parametros optimizados")
    print(popt)
    
    
    





frame =inicio()





ajuste=fraccional()

nuevo =comparacion_2(frame[0],frame[1])

otro = guru()
delta()


