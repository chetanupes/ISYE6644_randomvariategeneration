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
#Gamma
def rvgamma(n,lmbda=0.5, beta=0.5):
    X=[]
    U= uniform_rv(lower_limit=0, higher_limit=1, size=n)
    e=math.exp(1)
    if beta<1:
        b=(e+beta)/e
        for i in range(n-1):
            W=b*U[i]
            if W < 1:
                invbeta=1/beta
                Y=pow(W,invbeta)
                V=U[i+1]
                if V <= pow(e,-Y):
                    X.append(Y/lmbda)
            else:
                Y=-np.log((b-W)/beta)
                V=U[i+1]
                if V <= pow(Y,beta-1):
                    X.append(Y/lmbda)
    if beta>=1:
        a= pow(2*beta-1,-0.5)
        b=beta-np.log(4)
        c=beta+pow(a,-1)
        d=1 + np.log(4.5)
        for i in range(n-1):
            V=a*np.log(U[i]/(1-U[i]))
            Y=beta*pow(e,V)
            Z=pow(U[i],2)*U[i+1]
            W=b+(c*V)-Y
            if ((W+d-(4.5*Z)) >= 0):
                X.append(Y/lmbda)
            elif W >= np.log(Z):
                X.append(Y/lmbda)
    return X
#Poisson
def rvpoisson(n,lmbda=0.5):
    X1=[]
    e=math.exp(1)
    U=uniform_rv(lower_limit=0, higher_limit=1, size=n)
    a=pow(e,-lmbda)
    i=0
    while i < n:
        p=1
        X=-1
        while p > a:
            if i==n:
                break
            p=p*U[i]
            X=X+1
            i=i+1
        X1.append(X)
    return X1

st.title('ISYE-6644: Project (Random Variate Generator)')

#Team
st.title('Team 64')
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
Select_Method=st.sidebar.selectbox('Select a Model', ('Inverse Transform Method','Convolution Method', 'Acceptance-Rejection', 'Box Muller Method'))

#Method Selection

if Select_Method=='Inverse Transform Method':
    #Choosing a unifor dist
    st.sidebar.markdown('# Generating a Uniform')
    Lower=st.sidebar.number_input("Lower Limit",0)
    Upper=st.sidebar.number_input("Upper Limit",1)
    Size=st.sidebar.number_input("Size", 20)
    
    st.title('Inverse Transform Method')
    st.write('The inverse transform method is a general approach to generate random observations from any continuous probability distribution. This method involves finding the inverse of the cumulative distribution function (CDF) of the distribution you want to generate. You can then use the generated uniform random variable as the input to the inverse CDF to get the desired random variate. The inverse CDF can be computed analytically for some distributions or numerically using methods like bisection, Newton-Raphson, or Brents method. Inverse Transform Method follows the Inverse Transform Theorem, which states that let X be a continuous random variable with a cumulative distribution function (CDF) F(x), then F(X) ∼ Uniform (0, 1). Hence, we can produce the random variables of any distribution if we have uniform random variables and the CDF (in closed form) of that distribution using the formula below.')
    st.write('X= F-1(U), where F-1 is the inverse of F and U is a continuous uniform distribution (0,1)')
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
    
    #Choosing a unifor dist
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
        st.write('The triangular distribution provides a simplistic representation of the probability distribution when limited sample data is available. A triangular distribution is a continuous probability distribution that has a triangular shape. It is defined by three parameters: a, b, and c, where a is the minimum value, b is the maximum value, and c is the mode (the most common value).')
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
        st.write('The Erlang distribution is the distribution of a sum of k independent exponential variables. The Erlang distribution is a generalization of the exponential distribution. While the exponential random variable describes the time between adjacent events, the Erlang random variable describes the time interval between any event and the kth following event. It is used to model the number of events that occur in a fixed interval of time, given that the events occur independently and with a constant rate.')
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

if Select_Method=='Acceptance-Rejection':

    st.title('Acceptance-Rejection')
    st.write('The Acceptance-Rejection method is a general approach to generate random variates from any probability distribution. This method involves generating random samples from a simpler distribution that "covers" the target distribution and rejecting the samples that fall outside the target distribution. For example, you can generate uniform random variables in a rectangle that covers the target distribution and accept the samples that fall inside the target distribution. This method can be computationally expensive if the target distribution is complex or has a small acceptance rate.')

    #Choosing a unifor dist
    st.sidebar.markdown('# Generating a Uniform')
    Lower=st.sidebar.number_input("Lower Limit",0)
    Upper=st.sidebar.number_input("Upper Limit",1)
    Size=st.sidebar.number_input("Size", 20)
    
    st.sidebar.markdown("# Selecting RV Generator")
    Select_Dist=st.sidebar.selectbox('Select a Distribution', ('Gamma','Poisson'))


    if Select_Dist=='Gamma':
        st.title('{} Random Variable'.format(Select_Dist))
        st.write('Gamma Distribution is one of the distributions, which is widely used in the field of Business, Science and Engineering, in order to model the continuous variable that should have a positive and skewed distribution. Gamma distribution is a kind of statistical distributions which is related to the beta distribution. This distribution arises naturally in which the waiting time between Poisson distributed events are relevant to each other. ')

        st.sidebar.markdown('# Select Rate')
        lam=st.sidebar.number_input("Lambda",0.1, step=0.1)
        st.sidebar.markdown('# Select Beta')
        beta=st.sidebar.number_input("Beta",0.5, step=0.5)

        X=rvgamma(np.int64(Size),lam,beta)
        fig = px.histogram(X)
        fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

        #Plot!
        st.plotly_chart(fig, use_container_width=True)

    if Select_Dist=='Poisson':
            st.title('{} Random Variable'.format(Select_Dist))
            st.write('Poisson distribution is one of the important topics. It is used for calculating the possibilities for an event with the average rate of value. Poisson distribution is a discrete probability distribution. ')
            st.sidebar.markdown('# Select Rate')
            lam=st.sidebar.number_input("Lambda",0.1, step=0.1)
            
            X=rvpoisson(np.int64(Size), lam)
        
            fig = px.histogram(X)
            fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram: {} RV'.format(Select_Dist))

            #Plot!
            st.plotly_chart(fig, use_container_width=True,showlegend="false")


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
    fig.update_layout(showlegend=False,height=500, width=1300,barmode='group',title_text='Histogram Normal RV')

    #Plot!
    st.plotly_chart(fig, use_container_width=True)


#References
st.title('References')
st.write('[1] NumPy. (2023) from https://numpy.org/')
st.write('[2] Plotly. (2023) https://plotly.com/python/')
st.write('[3] Streamlit. (2023) https://streamlit.io/')
st.write('[4] Statistics By Jim. https://statisticsbyjim.com/')
st.write('[5] Wikipedia. https://en.wikipedia.org/wiki/Main_Page/')
st.write('[6] Medium. https://medium.com/')
st.write('[7] Law, A. M. (2015). Simulation modeling and analysis (5th Edition). New York: Mcgraw-Hill.')

st.write('Note: Kindly send your feedback, comments and recommendations to chetan.upes@gmail.com. Your feedback and comments are very valuable and will help us make this tool better and more useful.')
