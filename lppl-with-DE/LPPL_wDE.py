# -*- coding: utf-8 -*-
"""
Crash prediction using LPPL with Differential Evolution

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
import random as rand
#import peak_finding as pf
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
    return error_function(params[0],params[1],params[2],params[3],params[4],params[5],params[6])



# DE function
def DE(objf,lb,ub,dim,PopSize,iters):
   
    mutation_factor=0.5
    crossover_ratio=0.7
    stopping_func=None

    # convert lb, ub to array
    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]

    best = float("inf")
    leader_solution = []
    # initialize population
    population = []

    population_fitness = np.array([float("inf") for _ in range(PopSize)])

    for p in range(PopSize):
        sol = []
        for d in range(dim):
            d_val = rand.uniform(lb[d], ub[d])
            sol.append(d_val)

        population.append(sol)

    population = np.array(population)

    # calculate fitness for all the population
    for i in range(PopSize):
        fitness = objf(population[i, :])
        population_fitness[p] = fitness
        #s.func_evals += 1

        # is leader ?
        if fitness < best:
            best = fitness
            leader_solution = population[i, :]

    convergence_curve=np.zeros(iters)
   
    t = 0
    while t < iters:
        # should i stop
        if stopping_func is not None and stopping_func(best, leader_solution, t):
            break

        # loop through population
        for i in range(PopSize):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in  range(PopSize) if _ != i]
            id_1, id_2, id_3 = rand.sample(ids_except_current, 3)

            mutant_sol = []
            for d in range(dim):
                d_val = population[id_1, d] + mutation_factor * (population[id_2, d] - population[id_3, d])

                # 2. Recombination
                rn = rand.uniform(0, 1)
                if rn > crossover_ratio:
                    d_val = population[i, d]

                # add dimension value to the mutant solution
                mutant_sol.append(d_val)

            # 3. Replacement / Evaluation

            # clip new solution (mutant)
            mutant_sol = np.clip(mutant_sol, lb, ub)

            # calc fitness
            mutant_fitness = objf(mutant_sol)
            #s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness < population_fitness[i]:
                population[i, :] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness < best:
                    best = mutant_fitness
                    leader_solution = mutant_sol

        convergence_curve[t]=best
        if (t%1==0):
               print(['At iteration '+ str(t+1)+ ' the best fitness is '+ str(best)]);

        # increase iterations
        t = t + 1

    # return solution
    return leader_solution, best






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


#calculation of initial values of parameters

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


for ws in range(initial_window_size, len_x, 2):
    dim = 7
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
        # using DE for optimizing the parameters and fitting the LPPL
        opt_params, rmse = DE(objective_function, param_lower_bound, param_upper_bound, dim, 70, 100)
        rmse_tc_arr.append([opt_params[2], rmse])
        if rmse < min_rmse_for_this_window:
            min_rmse_for_this_window = rmse
            corresponding_tc = opt_params[2]
    print("Window size: ",window_size, " Param sets: ", len(params), " Minimum rmse: ",min_rmse_for_this_window)
    print("------------------------------------------------------------------------")
    if len(params) > 0:
        tc_rmse_arr_for_window.append([window_size, min_rmse_for_this_window, corresponding_tc])
        

rmse_tc_df = pd.DataFrame(rmse_tc_arr)
tc_rmse_window_df = pd.DataFrame(tc_rmse_arr_for_window)



rmse_tc_df.to_csv(r"E:\MTech\2nd_sem\ATSA\assignments\lppl\DE data\DE_rmse_tc.csv")
tc_rmse_window_df.to_csv(r"E:\MTech\2nd_sem\ATSA\assignments\lppl\DE data\DE_rmse_tc_window.csv")
        



'''
Following section is for plotting the relevant data as part of the requirements for report
'''
# plot for tc and closing price. The best value of tc is taken from the csv file created above
plt.figure(figsize=(10,6))
plt.plot(y)
plt.title("Closing price index and critical time")
plt.xlabel("Days")
plt.ylabel("Closing price")
plt.axvline(x=428.229449337481, color='r') # tc value taken from resulting file
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