import torch 
import torch.nn as nn 
import streamlit as st
import numpy as np 
import pandas as pd
import plotly.graph_objects as go 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

#---------

class LogisticRegression:

    def __init__(self ,lower_0, upper_0, sample_size_0, noise_0,lower_1, upper_1, sample_size_1, noise_1,w,b):

        # Parameters
        self.lower_0 = lower_0
        self.upper_0 = upper_0
        self.sample_size_0 = sample_size_0
        self.noise_0 = noise_0
        self.lower_1 = lower_1
        self.upper_1 = upper_1
        self.sample_size_1 = sample_size_1
        self.noise_1 = noise_1
        self.w = w
        self.b = b

        # made by attributes
        self.x0 = torch.linspace(lower_0, upper_0, sample_size_0) + torch.tensor([noise_0])
        self.x1 = torch.linspace(lower_1, upper_1, sample_size_1) - torch.tensor([noise_1])
        self.X = torch.cat((self.x0, self.x1), dim=0)
        self.y = torch.cat((torch.zeros(len(self.x0)), torch.ones(len(self.x1))), dim=0)

        # stand alone 
        self.inter_and_extrapolation = torch.linspace(-5,5,1000)
        self.possible_weights = torch.linspace(-5,25,100)
        self.possible_biases = torch.linspace(-5,5,100)       
        self.loss_fn = nn.BCEWithLogitsLoss()

        
        # made by stand alone
        self.weight_m , self.bias_m = torch.meshgrid(self.possible_weights,self.possible_biases,indexing='ij')
        self.weight_f = self.weight_m.flatten()
        self.bias_f   = self.bias_m.flatten()


#-----------------------------------------------------------------
# Math for loss_landscape, loss for classes & confusion matrix 

    def Loss(self):
        L = []

        for weight,bias in zip(self.weight_f,self.bias_f):
            z = (weight * self.X) + bias
            loss = self.loss_fn(z,self.y)
            L.append(loss)
        #------------------------
        L_f = torch.as_tensor(L)
        L_m = L_f.view(100,100)
        L_min = torch.argmin(L_f)
        min_index_w , min_index_b = np.unravel_index(L_min,(100,100))

        secret_weight = self.possible_weights[min_index_w]
        secret_bias   = self.possible_biases[min_index_b]

        return L_f,L_m,secret_weight,secret_bias

#-----------------------------------------------------------
    # Data points, sigmoid curve
    def generate_plot(self):
        scatter_class_0 = go.Scatter(
            x=self.x0,
            y=torch.zeros(len(self.x0)),
            mode='markers',
            marker=dict(color='purple'),
            name='class purple'
        )
        scatter_class_1 = go.Scatter(
            x=self.x1,
            y=torch.ones(len(self.x1)),
            mode='markers',
            marker=dict(color='orange'),
            name='class orange'
        )

        z = (self.w * self.inter_and_extrapolation) + self.b

        non_linear_line = go.Scatter(
            x= self.inter_and_extrapolation,
            y=torch.sigmoid(z),
            mode='lines',
            line={'color': 'rgb(27,158,119)'},
            name='model'
        )


        layout = go.Layout(
            xaxis=dict(
                title='X',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(205, 200, 193, 0.7)'
            ),
            yaxis=dict(
                title='Y/Y_hat',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(205, 200, 193, 0.7)'
            ),
            # height=500,
            # width=2600
        )
        figure = go.Figure(data=[scatter_class_0, scatter_class_1, non_linear_line], layout=layout)
        return figure
    
# -----------------------------------

    def loss_landscape(self,L_f,L_m,secret_weight,secret_bias):
        
        # landscape
            loss_landscape = go.Surface(
                    x = self.weight_m,
                    y = self.bias_m,
                    z = L_m,
                    name ='Loss function landscape',
                    opacity=0.7
                )

            # global
            Global_minima = go.Scatter3d(
                x = (secret_weight,),     
                y = (secret_bias,),
                z = (min(L_f),),
                mode = 'markers',
                marker = dict(color='yellow',size=10,symbol='diamond'),
                name = 'Global minima'
            )


            # ball
            z = (self.w * self.X ) +self.b   #forward pass
            loss = self.loss_fn(z,self.y)

            ball = go.Scatter3d(
                    x = (self.w,),
                    y = (self.b,),
                    z = (loss,),
                    mode = 'markers',
                    marker= dict(color='red'),
                    name = 'loss'
            )

            # layout 
            layout = go.Layout(
                 scene= dict(
                      xaxis = dict(title='weight'),
                      yaxis = dict(title='bias'),
                      zaxis = dict(title ='loss')
                 ),
                  legend=dict(
                          x=1.25,  # Position the legend to the right
                          y= 0.95,  # Vertically center the legend
                          bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent background
                          # bordercolor='black',
                          borderwidth=1
                      )
            )




            figure = go.Figure(data = [loss_landscape,Global_minima,ball],layout=layout)
            
            return figure
    

#------------------------------------------------------
# streamlit 
  

st.set_page_config(layout='wide')
st.title("Logistic Regression : Weight & Bias")
st.write('By : Hawar Dzaee')


with st.sidebar:

    st.subheader("Data Generation")
    sample_size_0_val = st.slider("sample size Class 0:", min_value= 2, max_value=12, step=1, value= 7)
    sample_size_1_val = st.slider("sample size Class 1:", min_value= 2, max_value=12, step=1, value= 9) 
    Noise = st.slider('Noise',min_value = 0.0, max_value = 1.0, step = 0.1, value = 0.2)

    # st.subheader('Threshold Selection')
    # threshold_val = st.slider('threshold',min_value=0.05,max_value=0.99,step=0.05,value=0.5)


    st.subheader("Adjust the parameter(s) to minimize the loss")
    w_val = st.slider("weight (w):", min_value=-4.0, max_value=14.0, step=0.1, value= 1.1) # first and all the widgets above 
    b_val = st.slider("bias (b):", min_value=-4.0, max_value=4.0, step=0.1, value= 1.1) # first and all the widgets above 



container = st.container()


with container:
 

    col1, col2 = st.columns([3,3])

    with col1:

        data = LogisticRegression(lower_0 = -2,upper_0 = 0, sample_size_0 = sample_size_0_val,noise_0 = Noise,
                                  lower_1 = 0, upper_1 = 2, sample_size_1 = sample_size_1_val, noise_1 = Noise,
                                  w = w_val,b=b_val) #second
        
        data.Loss() # third

        figure_1 = data.generate_plot() # fourth

        st.plotly_chart(figure_1, use_container_width=True)



      
        st.write('-------------')

    L_f,L_m,secret_weight,secret_bias = data.Loss() # fifth
    with col2:
       figure_2 = data.loss_landscape(L_f,L_m,secret_weight,secret_bias) # seventh
       st.plotly_chart(figure_2,use_container_width=True)
       st.write('---------------')
       


