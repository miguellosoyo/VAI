# Importar librerías de trabajo
import os
import random
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_echarts import JsCode
from streamlit_echarts import st_echarts

# Definir una función para obtener la distribución de pesos para una lista de variables
def weighted(elements:list, step:float=10, start:int=7, end:int=11, threshold:float=0.32, seed:bool=True):

  # # Evaluar si el usuario requiere de una semilla para replicar los resultados
  # if seed:
  #   random.seed(123)

  # Iterar para asignar un valor a cada elemento de la lista y hacer ingrementos progresivos
  weights = [] # Definir una lista vacía para almacenar los pesos
  i=1 # Definir contador
  for i in range(len(elements)-1):    

    # Calcular factor
    weight = round(random.randrange(start, end)/100,2)

    # Evaluar si es el primer registro
    if weights==[]:
      
      # Calcular el peso correspondiente y actualizar la lista
      weights.append(weight)

      # Incrementar contador
      i+=1
    
    # Evaluar que la suma sea menor a 1
    else:

      # Recalcular los factores de inicio y fin para los valores
      start+=step
      end+=step 

      # Calcular el peso correspondiente y actualizar la lista
      weights.append(weight)

  # Integrar el valor del último activo intangible
  weights.append(round(1-sum(weights),2))

  # Devolver la lista de pesos
  return sorted(weights, reverse=True)

# Crear una función para obtener un DataFrame con las proporciones respecto a los ingresos, de los activos intangibles de cada actividad
def intangible_df(activities:list, intangibles:list):

  # Repetir la actividad para el número correspondiente activos intangibles
  activities = [[x]*len(intangible) for x, intangible in zip(activities,intangibles)]

  # Obtener la asignación de pesos para cada activo intangible
  weighted_participation = []
  for intangible in intangibles:
    weighted_participation.append(weighted(intangible))

  # Transformar los elementos del DataFrame a arryas
  activities = np.ndarray.flatten(np.array(activities))
  intangibles = np.ndarray.flatten(np.array(intangibles))
  weighted_participation = np.ndarray.flatten(np.array(weighted_participation))

  # Integrar la información en un DataFrame
  df = pd.DataFrame({'Actividad':activities, 'Intangibles':intangibles, 'Proporciones':weighted_participation})

  return df

# Definir una función que ilumine una fila sí y otra no de un color en específico
def highlight(x):

  # Definir los colores para las filas impares y pares
  c1 = 'background-color: #dedede'
  c2 = 'background-color: white'

  # Definir un DataFrame con el mapeo de los colores
  df1 = pd.DataFrame('', index=x.index, columns=x.columns)
  df1.loc[df1.index%2!=0, :] = c2
  df1.loc[df1.index%2==0, :] = c2

  return df1  

# Crear una función para calcular la TIR
def irr(values, guess=0.1, tol=1e-12, maxiter=100):
    
  # Convertir los valores integrados a un vector de una dimensión
  values = np.atleast_1d(values)
  
  # Evaluar que sea un vector unidimensional
  if values.ndim != 1:
      raise ValueError("Cashflows must be a rank-1 array")

  # Evaluar que los valores sean mayores a 0
  same_sign = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
  if same_sign:
      return np.nan

  # Crear una función polinómica con los valores
  npv_ = np.polynomial.Polynomial(values)
  
  # Aplicar su derivada
  d_npv = npv_.deriv()
  x = 1 / (1 + guess)

  # Iterar sobre las derivadas hasta obtener la TIR
  for _ in range(maxiter):
      x_new = x - (npv_(x) / d_npv(x))
      if abs(x_new - x) < tol:
          return 1 / x_new - 1
      x = x_new

  # Devolver valor nulo en caso de no hallar la TIR
  return np.nan

# Integrar un sidebar para que el usuario pueda ingresar información
with st.sidebar:

  # Importar lista de actividades, activos intangibles y proporciones
  df_prop = pd.read_csv('https://raw.githubusercontent.com/miguellosoyo/VAI/main/data/Proporcio%CC%81n%20de%20Intangibles%20por%20Actividad%20Econo%CC%81mica.csv')

  # Integrar un subtitulo para la sección
  st.subheader('Sección de Datos Empresariales')

  # Integrar campos de texto para capturar el RFC, Razón Social, Actividad Económica, Ingresos Anuales
  rfc = st.text_input('Ingrese el RFC de la Empresa')
  business_name = st.text_input('Ingrese la Razón Social de la Empresa')
  activity = st.selectbox('Elija la Actividad Económica que Realiza su Empresa', options=df_prop['Actividad'].unique())
  income = st.number_input('Ingrese el Ingreso Anual más Reciente', value=int(1e7), step=int(1e3))
  
  # Importar lista de actividades, margen neto y proporción de activos intangibles sobre ventas
  df_eco = pd.read_csv('https://raw.githubusercontent.com/miguellosoyo/VAI/main/data/Informacio%CC%81n%20Econo%CC%81mica%20por%20Actividad.csv')

  # Integrar un subtitulo para la sección
  st.subheader('Sección de Datos Económicos')
  st.write(df_eco.columns)

  # Integrar campos de texto para capturar/modificar los niveles de margen neto, inflación e ISR
  net_margin = st.number_input('Margen Neto de la Actividad Económica', min_value=0., max_value=100., 
                               value=(df_eco[df_eco['Actividad']==activity]['Margen Neto'][0])*100, step=1.)/100
  inf_rate = st.number_input('Inflación Anual', min_value=0, max_value=100, value=10, step=1)/100
  tax_rate = st.number_input('Tasa de ISR', min_value=0, max_value=50, value=30, step=1)/100
  int_rate = st.number_input('Participación de Intangibles sobre Ventas', min_value=0., max_value=100., 
                                          value=(df_eco[df_eco['Actividad']==activity]['Intangibles'][0])*100, step=1.)/100

# Definir la información base
raw_data = {'RFC':[rfc],
            'RAZÓN SOCIAL':[business_name],
            'ACTIVIDAD':[activity],
            'INGRESO ANUAL':[income],
            'MARGEN NETO MIN':[net_margin*0.79],
            'MARGEN NETO PROM':[net_margin],
            'MARGEN NETO MAX':[net_margin*1.21],
            'PARTICIPACIÓN INTANGIBLE MIN':[int_rate*0.91],
            'PARTICIPACIÓN INTANGIBLE PROM':[int_rate],
            'PARTICIPACIÓN INTANGIBLE MAX':[int_rate*1.09],
            'PARTICIPACIÓN INTANGIBLE REM':[1-int_rate],
            'EBT':[round(((income*net_margin)/(1-tax_rate))/1e6, 0)],
            }
df = pd.DataFrame(raw_data)

# Calcular las tablas de flujos de efectivo sin VAI
out_vai = pd.DataFrame({'Utilidad antes de Impuestos':[raw_data['EBT'][0]*((1+inf_rate)**i) for i in range(0,7)], 
                        'ISR':[i for i in range(0,7)], 'Utilidad Neta':[i for i in range(0,7)], 'Costo Fiscal':[0 if i==0 else np.nan for i in range(0,7)]}, 
                       index=[f'A{i}' for i in range(0,7)]).T
out_vai.loc['ISR',:] = out_vai.loc['Utilidad antes de Impuestos', :].multiply(tax_rate)
out_vai.loc['Utilidad Neta',:] = out_vai.loc['Utilidad antes de Impuestos', :].sub(out_vai.loc['ISR', :])
out_vai.loc['Costo Fiscal', 'A0'] = out_vai.loc['ISR',:].sum()
out_vai = out_vai.round(2).reset_index() # Redondear a una cifra decimal y reiniciar índice
out_vai.rename(columns={'index':'Costo fiscal Sin VAI'}, inplace=True) # Renombrar primera columna

# Aplicar el formato definido en el caso respectivo, y esconder el índice de números consecutivos
out_vai = out_vai.style.apply(highlight, axis=None).set_properties(**{'font-size': '10pt', 'font-family': 'monospace', 'border': '', 'width': '110%'}).format(format)
# out_vai = out_vai.style.set_properties(**{'font-size': '10pt', 'font-family': 'monospace', 'border': '', 'width': '110%'}).format(format)

# Definir las propiedades de estilo para los encabezados
th_props = [
            ('font-size', '12pt'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', 'white'),
            ('background-color', '#404040'),
            ('width', '70%'),
            ]

# Definir las propiedades de estilo para la información de la tabla
td_props = [
            ('font-size', '8pt'),
            ('width', '110%'),
            ('text-align', 'center'),
            ('border', '0.1px solid white'),
            ]

# Integrar los estilos en una variable de estilos
styles = [
          dict(selector='th', props=th_props),
          dict(selector='td', props=td_props),
          {'selector':'.line', 'props':'border-bottom: 0.5px solid #000066'},
          {'selector':'.False', 'props':'color: white'},
          {'selector':'.Cost', 'props':[('font-weight', 'bold'), ('background-color', '#f2f2f2')]},
          {'selector':'.w', 'props':[('background-color','white'), ('color','black')]},
          ]

# Integrar líneas si el índice corresponde a una posición de la tabla
cell_border = pd.DataFrame([['line']*len(x) if i==1 or i==2 else ['']*len(x) for i, x in out_vai.data.iterrows()], columns=out_vai.data.columns)
cell_fiscal_cost = pd.DataFrame([x.notnull().astype(str).replace('True', 'w').tolist() if i==0 else (x.notnull().astype(str).replace('True', 'Cost').tolist() if i==3 
                                                                                                     else ['False']*len(x)) for i, x in out_vai.data.iterrows()], columns=out_vai.data.columns)

# Aplicar formatos sobre las clases definidas
out_vai = out_vai.set_table_styles(styles).set_td_classes(cell_fiscal_cost).set_td_classes(cell_border)

# Definir formato CSS para eliminar los índices de la tabla, centrar encabezados, aplicar líneas de separación y cambiar tipografía
hide_table_row_index = """
                        <style>
                        tbody th {display:none;}
                        .blank {display:none;}
                        .col_heading {font-family: monospace; border: 3px solid white; text-align: center !important;}
                        </style>
                      """
hide_table_row_index = """
                        <style>
                        tbody th {display:none;}
                        .blank {display:none;}
                        .col_heading {font-family: monospace; text-align: center !important;}
                        </style>
                      """

# Integrar el CSS con Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

# Mostrar información financiera
st.subheader(f'''
            Flujos de Efectivo Sin Valuación de Activos Intangibles
            ''')

# Integrar el DataFrame a la aplicación Web
st.markdown(out_vai.to_html(), unsafe_allow_html=True)
# st.write(out_vai)

# Insertar una nota al pie de la tabla
st.caption(f'Resultados en millones de pesos, estimados con base en información financiera de la actividad económica.')
