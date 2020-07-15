# -*- coding: utf-8 -*-
"""
Crash prediction using LPPL with Grey Wolf Optimizer

Notes:  
1. This file uses dataset which has some manual effort to create a column named 'ti' since the dataset is small in size hence does not 
   include the code for putting the column. The column was added using microsoft excel. 
2. peak_finding.py file should be in the same directory as this file and should be built before running this file. 
3. The locations of data and result are system dependent and should be changed as per needed. 
4. This is a optimization problem and metaheuristic in nature so, several runs are required for getting a good result. 
5. This is developed using Spyder IDE in Anaconda Development Environment for Python which allows running methods and lines of scripts
   in any order. Sequence does not matter except for reference error. Please consider using the same for convenience. 
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from peak_finding import peak_finding



# function to evaluate LPPL
def LPPL(t, A, B, tc, m, C, omega, phi):
    lppl_val = A + B*((tc-t)**m)*(1 + C*(np.cos(omega * np.log(tc-t) + phi)))
    return lppl_val;


# function to calculate RMSE
def error_function(A, B, tc, m, C, omega, phi):
    observed_values = np.log(y_train)
    predicted_values = LPPL(X_train, A, B, tc, m, C, omega, phi)
    diff_sqr = (predicted_values - observed_values)**2
    total = np.sum(diff_sqr)
    total = total/len(observed_values)
    rmse = math.sqrt(total)
    return rmse


def objective_function(params):
#    print("params ",params[1])
    return error_function(params[0],params[1],params[2],params[3],params[4],params[5],params[6])


# GWO function
def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    
    Convergence_curve=np.zeros(Max_iter)
    # s=solution()  #delete

     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")
    
    # timerStart=time.time()   #delete
    # s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")  #delete
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
            print("Fitness: ",fitness)
            
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=Alpha_score  # Update beta
                Beta_pos=Alpha_pos.copy()
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score):                 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
        
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
            
        
        
        Convergence_curve[l]=Alpha_score;

        if (l%1==0):
               print(['At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)]);

    return Alpha_pos, Alpha_score


'''
    Implementation of LPPL for crash prediction
    
'''

# linear regression for determining initial values of A, B

def A_B_initial_values(ind_var, dep_var):
    ind_var = ind_var.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(ind_var, dep_var)
    param_A = regressor.intercept_
    param_B = regressor.coef_[0]
    return param_A, param_B


# this function returns array of initial values of all params using peak finding and above A_B_initial_values function

def tc_omega_phi_initial_values(window_size):
    #finding peaks for the given window size
    
    peaks_arr = peak_finding(y_train, window_size)
    x_peaks = [ x+1 for x,y in peaks_arr ]

    init_all_params_arr = []

    peaks_arr_len = len(peaks_arr)
    for i in range(0, peaks_arr_len-2):
        j = i+1
        k = i+2
        pi = x_peaks[i]
        pj = x_peaks[j]
        pk = x_peaks[k]
        temp_ro = (pj-pi)/(pk-pj)
        if (temp_ro <= 1):
            continue;
        param_tc = (temp_ro*pk - pj)/(temp_ro - 1)
        param_omega = (2 * math.pi)/(np.log(temp_ro))
        if (param_tc <= pk or param_tc <= X_train[len(X_train)-1]):
            continue;
        param_phi = math.pi - param_omega * np.log(param_tc - pk)
        param_m = 1 # this is beta
        param_C = 0
        ind_var_arr = param_tc - X_train
        param_A, param_B = A_B_initial_values(ind_var_arr, y_train_log)
        

        print("A: ",param_A," B: ",param_B," tc: ",param_tc," m: ",param_m," C: ",param_C," omega: ",param_omega," phi: ",param_phi)
        init_all_params_arr.append([param_A, param_B, param_tc, param_m, param_C, param_omega, param_phi])
        
    return init_all_params_arr
        



## From this point on, every block of script should be executed on selection.


col_list = ['ti', 'Close']
datadf = pd.read_csv(r"E:\MTech\2nd_sem\ATSA\assignments\lppl\dataset.csv", usecols = col_list)

X = datadf['ti'].to_numpy()
y = datadf['Close'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
y_train_log = np.log(y_train)

      
# try with several window sizes


# loops through several window sizes and note rmse and tc
initial_window_size = 10
len_x = len(X_train)

tc_rmse_arr_for_window = []
rmse_tc_arr = []
all_params_arr = []


for ws in range(initial_window_size, len_x, 1):
    dim = 7
    GWO_search_agent_num = 50
    window_size = ws
    params = tc_omega_phi_initial_values(window_size)
    min_rmse_for_this_window = 999999999.99
    corresponding_tc = 0
    for i in range(0, len(params)):
        param_lower_bound = params[0]
        param_upper_bound = [0]*7
        param_upper_bound[0] = 2 * abs(param_lower_bound[0])    # A
        param_upper_bound[1] = max(2 * abs(param_lower_bound[1]), 2)    # B
        param_upper_bound[2] = abs(param_lower_bound[2]) + 300    # tc
        param_upper_bound[3] = 2    # m or beta
        param_upper_bound[4] = 1    # C
        param_upper_bound[5] = 2 * abs(param_lower_bound[5])    # omega
        param_upper_bound[6] = 2*math.pi    # phi
        # using GWO for optimizing the parameters and fitting the LPPL
        opt_params, rmse = GWO(objective_function, param_lower_bound, param_upper_bound, dim, GWO_search_agent_num, 100)
        rmse_tc_arr.append([opt_params[2], rmse])
        all_params_arr.append([opt_params[0],opt_params[1],opt_params[2],opt_params[3],opt_params[4],opt_params[5],opt_params[6],rmse])
        if rmse < min_rmse_for_this_window:
            min_rmse_for_this_window = rmse
            corresponding_tc = opt_params[2]
    print("Window size: ",window_size, " Param sets: ", len(params), " Minimum rmse: ",min_rmse_for_this_window)
    print("------------------------------------------------------------------------")
    if len(params) > 0:
        tc_rmse_arr_for_window.append([window_size, min_rmse_for_this_window, corresponding_tc])
        

rmse_tc_df = pd.DataFrame(rmse_tc_arr)
tc_rmse_window_df = pd.DataFrame(tc_rmse_arr_for_window)

rmse_tc_df.to_csv(r"E:\MTech\2nd_sem\ATSA\assignments\lppl\GWO data\GWO_rmse_tc.csv")
tc_rmse_window_df.to_csv(r"E:\MTech\2nd_sem\ATSA\assignments\lppl\GWO data\GWO_rmse_tc_window.csv")
        



'''
Following section is for plotting the relevant data as part of the requirements for report
'''
# plot for tc and closing price. The best value of tc is taken from the csv file created above
plt.figure(figsize=(10,6))
plt.plot(y)
plt.title("Closing price index and critical time")
plt.xlabel("Days")
plt.ylabel("Closing price")
plt.axvline(x=443.259195259618, color='r') # tc value taken from resulting file
plt.axvline(x=419.300960887724, color='r') # tc value taken from resulting file
plt.show()

# plot of training data
plt.figure(figsize=(10,6))
plt.plot(y_train)
plt.title("Closing price index")
plt.xlabel("Days")
plt.ylabel("Closing price")

# plot of whole data
plt.figure(figsize=(10,6))
plt.plot(y)
plt.title("Closing price index")
plt.xlabel("Days")
plt.ylabel("Closing price")









