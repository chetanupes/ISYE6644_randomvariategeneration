from functools import total_ordering
from os import write
import pandas as pd
import numpy as np
#from jinja2 import escape
import joblib
import math

#Streamlit
import streamlit as st
st.set_page_config(layout="wide")

#Plotting
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Warnings
import warnings
import time
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Widgets libraries
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive

#############################################################################################################################################
#Function to create a uniform distribution using the LCG
size=10000
U=np.zeros(size)
seed=123456789
mod=2**31-1
x=(16807*seed+1)%mod
U[0]=(16807*seed+1)%mod
for i in range(0,size):
    x=(x*16807+1)%mod
    U[i]=x/mod

#Creating random seed
def random_time_seed():
    t=time.perf_counter()
    seed=int(10**9*float(str(t-int(t))[0:]))
    return seed



#LCG to generate uniforms
def lcg_rv(size=1):
    U=np.zeros(size)
    mod=2**31-1
    seed=random_time_seed()
    x=(seed*16807)%mod
    U[0]=x%mod
    for i in range(0,size):
        x=(x*16807)%mod
        U[i]=x/mod
    return U

#Generating Uniforms
def uniform_rv(lower_limit=0, higher_limit=1,size=1):
    return (lower_limit+(higher_limit-lower_limit)*lcg_rv(size=size))

#Bernoulli
def rvbern(p,n):
    X=[]
    U=uniform_rv(lower_limit=0, higher_limit=1,size=n)
    for i in U:
        if(i<=1-p):
            X.append(0)
        else:
            X.append(1)
    return X


st.title('ISYE-6644: Project (Random Variate Generator)')

#Team
st.title('Team Members')
st.write('1. Shameema Shahul Hameed')
st.write('2. Chetan Tewari')

#st.title('Project Title: Random Variate Generator')
st.title('Objective/Problem')
st.write('In probability and statistics, random variables are a fundamental concept. They are important because they allow us to formalize and study the behavior of uncertain or random events, which occur frequently in many real-world applications')
st.write('Here are some reasons why random variables are essential:')

st.write('1. Modeling uncertainty')
st.write('2. Predictive modeling')
st.write('3. Statistical inference')
st.write('4. Optimization')
st.write('The core objective of this project is to generate random variates (observations) from scratch which involves generating uniform random variables, choosing a distribution for which random variates are to be generated, and finally applying different methods to transform the uniform random variables into the desired distribution. The methods we are planning to use in thisproject include Inverse Transform, Convolution, Acceptance/Rejection, Box Muller, and Composition. The project envisions creating a library of random variate generation routines using Python. The project will include library routines for creating random variates for the following discrete and continuous distributions:')

# Selecting the method for analysis   
st.sidebar.markdown("# Selecting RV Generator Method")
Select_Method=st.sidebar.selectbox('Select a Model', ('Inverse Transform Method','Convolution Method', 'Box Muller Method'))

#Method Selection

if Select_Method=='Inverse Transform Method':
    #Choosing a unifor dist
    st.sidebar.markdown('# Generating a Uniform')
    Lower=st.sidebar.number_input("Lower Limit",0)
    Upper=st.sidebar.number_input("Upper Limit",1)
    Size=st.sidebar.number_input("Size", 20)
    
    st.title('Inverse Transform Method')
    st.write('The inverse transform method is a general approach to generate random observations from any continuous probability distribution. This method involves finding the inverse of the cumulative distribution function (CDF) of the distribution you want to generate. You can then use the generated uniform random variable as the input to the inverse CDF to get the desired random variate. The inverse CDF can be computed analytically for some distributions or numerically using methods like bisection, Newton-Raphson, or Brents method. Inverse Transform Method follows the Inverse Transform Theorem, which states that let X be a continuous random variable with a cumulative distribution function (CDF) F(x), then F(X) ∼ Uniform (0, 1). Hence, we can produce the random variables of any distribution if we have uniform random variables and the CDF (in closed form) of that distribution using the formula below.')
    st.write('X= F-1(U), where F-1 is the inverse of F and U is a continuous uniform di\hw2web\external-accounts\widget\embeddedTmcWidgetContainer.xhtmlstribution (0,1)')
    st.write('Pseudo code for generating random variables using Inverse Transform Method from U (0,1):')
    st.write('1. Generate the Uniform Random variables U: U1, U2, U3, …, Un using the uniform_rv function as shown in the code above.')
    st.write('2. Calculate the F-1 of F')
    st.write('3. Generate the independent random variate x1, x2, x3, …, xn of random variable X, whereeach xi = F-1 (Ui) for i = 1, 2, ..., n')

    st.sidebar.markdown("# Selecting RV Generator")
    Select_Dist=st.sidebar.selectbox('Select a Distribution', ('Bernoulli','Exponential','Weibull','Geometric'))


    if Select_Dist=='Bernoulli':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('A binomial distribution can be thought of as simply the probability of a SUCCESS or FAILURE outcome in an experiment or survey that is repeated multiple times. The binomial is a type of distribution that has two possible outcomes (the prefix “bi” means two, or twice). For example, a coin toss has only two possible outcomes: heads or tails and taking a test could have two possible outcomes: pass or fail.')

        st.sidebar.markdown('# Select probability')
        p=st.sidebar.number_input("p",min_value=0.01,max_value=1.0, step=0.1)
        X=rvbern(p,np.int64(Size))

        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True)

    if Select_Dist=='Exponential':
            st.title('{} Random Variable'.format(Select_Dist))
            st.write('Exponential distribution is a continuous probability distribution that often concerns the amount of time until some specific event happens. It is a process in which events happen continuously and independently at a constant average rate. The exponential distribution has the key property of being memoryless. The exponential random variable can be either more small values or fewer larger variables. For example, the amount of money spent by the customer on one trip to the supermarket follows an exponential distribution.')
            st.sidebar.markdown('# Select Rate')
            lam=st.sidebar.number_input("Lambda",0.1, step=0.1)
            U=uniform_rv(lower_limit=Lower, higher_limit=Upper,size=np.int64(Size))
            X=-(1/lam)*(np.log(U))
        
            fig = px.histogram(X)
            fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

            #Plot!
            st.plotly_chart(fig, use_container_width=True,showlegend="false")

    if Select_Dist=='Weibull':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('The Weibull Distribution is a continuous probability distribution used to analyse life data, model failure times and access product reliability. It can also fit a huge range of data from many other fields like economics, hydrology, biology, engineering sciences. It is an extreme value of probability distribution which is frequently used to model the reliability, survival, wind speeds and other data.')
        st.sidebar.markdown('# Select Rate')
        lam=st.sidebar.number_input("Lambda",0.1, step=0.1)
        st.sidebar.markdown('# Select Beta')
        beta=st.sidebar.number_input("Beta",0.5, step=0.5)
        U=uniform_rv(lower_limit=Lower, higher_limit=Upper,size=np.int64(Size))
        X=-(1/lam)*(np.log(U))**1/beta
        
        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True)

    elif Select_Dist=='Geometric':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('A geometric distribution is defined as a discrete probability distribution of a random variable “x” which satisfies some of the conditions. The geometric distribution conditions are')
        st.write('1. A phenomenon that has a series of trials')
        st.write('2. Each trial has only two possible outcomes – either success or failure')
        st.write('3. The probability of success is the same for each trial')
        st.sidebar.markdown('# Select probability')
        p=st.sidebar.number_input("p",min_value=0.01,max_value=1.0, step=0.1)
        U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=np.int64(Size))
        X=[]
        for i in U:
            X.append(math.ceil((np.log(i))/(np.log(1-p))))

        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True,showlegend="false")
    
    
if Select_Method=='Convolution Method':
   
    #Choosing a uniform dist
    st.sidebar.markdown('# Generating a Uniform')
    Lower=st.sidebar.number_input("Lower Limit",0)
    Upper=st.sidebar.number_input("Upper Limit",1)
    Size=st.sidebar.number_input("Size", 20)
    
    Select_Dist=st.sidebar.selectbox('Select a Distribution', ('Binomial','Triangular','Erlang'))

    st.title('Convolution Method')
    st.write('The convolution method is nothing but adding random variables to generate random observations. Some examples where we can use convolution method to generate random variables are Binomial - where we can add up the random variables from the Bernoulli distribution and Erlang distribution - where we can add up the random variables from exponential distribution.')


    if Select_Dist=='Binomial':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('A binomial distribution is simply the probability of a SUCCESS or FAILURE outcome in an experiment or survey that is repeated multiple times. The binomial is a type of distribution that has two possible outcomes (the prefix “bi” means two, or twice). For example, a coin toss has only two possible outcomes: heads or tails and taking a test could have two possible outcomes: pass or fail.')
        st.sidebar.markdown('# Select probability')
        p=st.sidebar.number_input("p",min_value=0.01,max_value=1.0, step=0.1)
        st.sidebar.markdown('# Select Number of Trials')
        k=st.sidebar.number_input("k", 20)
        
        X=[]
        for j in range(0,np.int64(k)):
            U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=np.int64(Size))
            Y=(U<=p).astype(int)
            X.append(np.sum(Y))

        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True)
    if Select_Dist=='Triangular':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('The triangular distribution provides a simplistic representation of the probability distribution when limited sample data is available.')
        X=[]
        for i in range(0,np.int64(Size)):
            U1=U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=1)
            U2=U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=1)
            X.append((U1+U2)[0])

        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True)

    elif Select_Dist=='Erlang':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('The Erlang distribution is the distribution of a sum of k independent exponential variables. The Erlang distribution is a generalization of the exponential distribution. While the exponential random variable describes the time between adjacent events, the Erlang random variable describes the time interval between any event and the kth following event.')
        st.sidebar.markdown('# Select Rate')
        lam=st.sidebar.number_input("Lambda",0.1, step=0.1)
        n=st.sidebar.number_input("Erlang-k", 1)
        X=0
        for i in range(0,np.int64(n)):
            U=U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=np.int64(Size))
            X+=(-(1/lam)*(np.log(U)))
        
        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True)


if Select_Method=='Box Muller Method':
    st.title('Normal Random Variable')
    st.write('The Box–Muller transform, by George Edward Pelham Box and Mervin Edgar Muller, is a random number sampling method for generating pairs of independent, standard, normally distributed (zero expectation, unit variance) random numbers, given a source of uniformly distributed random numbers.')
    st.sidebar.markdown('# Generating a Uniform')
    Lower=st.sidebar.number_input("Lower Limit",0)
    Upper=st.sidebar.number_input("Upper Limit",1)
    Size=st.sidebar.number_input("Size", 20)
    st.sidebar.markdown('# Mean & Standard Deviation')
    mu=st.sidebar.number_input("Mean",0)
    sigma=st.sidebar.number_input("SD", 1)

    X=[]
    for i in range(0,np.int64(Size)):
        U1=U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=1)
        U2=U=uniform_rv(lower_limit=Upper, higher_limit=Lower,size=1)
        z=np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
        z=z*sigma+mu
        X.extend(z)

    fig = px.histogram(X)
    fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: Normal RV')

    #Plot!
    st.plotly_chart(fig, use_container_width=True)


#References
st.title('References')
st.write('[1] Keller, C.; Glück, F.; Gerlach, C.F.; Schlegel, T. Investigating the Potential of Data Science Methods for Sustainable Public Transport. Sustainability 2022, 14, 4211.')
st.write('[2] Burr, T., Merrifield, S., Duffy, D., Griffiths, J., Wright, S., Barker, G., 2008. Reducing Passenger Rail Delays by Better Management of Incidents. Stationery Office, London. ')
st.write('[3] Preston, J., Wall, G., Batley, R., Ibáñez, J.N., Shires, J., 2009. Impact of delays on passenger train services. Transport. Res. Rec.: J. Transport. Res. Board 2117 (1), 14–23.')
st.write('[4] Fredrik Monsuur, Marcus Enoch, Mohammed Quddus, Stuart Meek, Modeling the impact of rail delays on passenger satisfaction, Transportation Research Part A: Policy and Practice, Volume 152, 2021, Pages 19-35, ISSN 0965-8564.')
st.write('[5] Nils O.E. Olsson, Hans Haugland, Influencing factors on train punctuality—results from some Norwegian studies, Transport Policy, Volume 11, Issue 4, 2004, Pages 387-397, ISSN 0967-070X.')
