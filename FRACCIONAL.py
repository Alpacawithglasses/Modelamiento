import os
import pandas as pd
import matplotlib.pyplot as plt



import csv
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

from math import exp
from math import pi


from scipy.optimize import curve_fit
from lmfit import Model, Parameters

def inicio():
    datos = pd.read_csv("datos2.csv")
    print(datos.info()) ##Muestra el tipo de datos
    print(datos.head())## Muestra los primeros 5 datos (encabezado)
    print(datos.describe())##descripcion estadistica de los datos
    
    x= datos.tiempo.values.reshape(-1,1)
    y = datos.hr.values.reshape(-1,1)  ####Redimenciono altiro los valores
    #Para poder utilizrlos en los modelos
    
    #x = np.linspace(0,450,40)#### Crea un arreglo donde los parametros son (punto de partida, punto final, saltos)
    #plt.title("MOISTURE RATIO VS TIME (WITH TIME OF MINUTES")
    #fig,axes = plt.subplots()
    ##axes.scatter(datos["tiempo"],datos["hr"]) Entrega los datos como puntos
    plt.figure(figsize=(9,5))###Se le dalas simenciones al gradico, (ancho, alto)
    #Face color cambia el color de la ventana
    
    plt.plot(x,y,"r") #Cambia el color de la linea
    plt.title("Humedad vs tiempo (minutos)")
    plt.ylabel('Contenido de Humedad')
    plt.xlabel('Tiempo en minutos')
    plt.legend(["contenido de humedad"],loc =3) #por defecto coloca la leyenda arriba
    #Si no se colocan corchetes no se muestra el mensaje entero
    plt.show()
    print()
    return x,y
   






def ecuacion(x,difusividad):
    return (8/pi**2)*exp(-difusividad*x*(pi/0.004)**2)




def comparacion_1(difusividad,x_convertida,y):
   
    y_convertida2 = []
    for elemento in x_convertida:
       
        valor = ecuacion(elemento,difusividad)
        y_convertida2.append(valor)
   
     
  
        
        
  
    
    plt.figure(figsize=(9,5))
    plt.plot(x_convertida,y,"r",x_convertida,y_convertida2) #Cambia el color de la linea
    plt.title("Datos reales vs datos modelados")
    plt.ylabel('Contenido de Humedad')
    plt.xlabel('Tiempo (segundos)')
    plt.legend(["contenido de humedad real","contenido de humedad modelado"],loc =1) #por defecto coloca la leyenda arriba
    #Si no se colocan corchetes no se muestra el mensaje entero
    plt.show()
    
    
    
def ecuacion_2(x,difusividad2,alfa):
    return -difusividad2*(9.869604401/0.000016)*x**alfa



def fraccional():
    datos = pd.read_csv("datos.csv")
    x= (datos.tiempo.values.reshape(-1,1))
    y = np.log(datos.hr.values.reshape(-1,1)*(9.869604401/8))
    #print(x)
    #print(y)
    model = Model(ecuacion_2,independent_vars=['x'], param_names=["difusividad2", "alfa"])

    params = Parameters()
    params.add("difusividad2", value= 0.0000000000001,min =0,max = 0.0001)
    params.add("alfa", value=0.00000000000001,min=1,max = 2.6905)
    
    #R maximo alcanzado = 0.99683362
    #df = 4.1800e-17
    #alfa = 2.64370597
    #0.0000000000001

    result = model.fit(data=y, params=params, x=x)
    #print(model)


    #x = np.linspace(0,22000,410)
    #fig,axes = plt.subplots()
    #axes.scatter(data["tiempo"],data["hr"])
    #axes.plot(x,funcion(x,-1.26811894e-11,1.66027663e+10,-4.09643193e-03,-6.06639361e+09))
    #plt.show()
        
    print(result.fit_report())
    

    
    
    
    
    

    
   
    difusividad2 = 0
    alfa =0
   
    return difusividad2,alfa










def ecuacion_3(x,difusividad,alfa):
    return (8/pi**2)*exp(-difusividad*x**alfa*(pi/0.004)**2)




def comparacion_2(difusividad2,alfa,x_convertida,y):
    difusividad2 = 4.1800e-17

    alfa = 2.64370597

    
    y_convertida3 = []
    for elemento in x_convertida:
       
        valor = ecuacion_3(elemento,difusividad2,alfa)
        y_convertida3.append(valor)
   
     
  
        
        
  
    
    plt.figure(figsize=(9,5))
    plt.plot(x_convertida,y,"r",x_convertida,y_convertida3,"") #Cambia el color de la linea
    plt.title("Datos reales vs datos modelados de modo fraccional ")
    plt.ylabel('Contenido de Humedad')
    plt.xlabel('Tiempo  (segundos)')
    plt.legend(["contenido de humedad real","contenido de humedad modelado"],loc =1) #por defecto coloca la leyenda arriba
    #Si no se colocan corchetes no se muestra el mensaje entero
    plt.show()
    
    





frame =inicio()


    