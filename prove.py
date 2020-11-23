import numpy as np
import pandas as pd

df_x = pd.read_csv("./data/logistic_x.txt", sep="\ +", names=["x1","x2"], header=None, engine='python')
df_y = pd.read_csv('./data/logistic_y.txt', sep='\ +', names=["y"], header=None, engine='python')
df_y = df_y.astype(int)
df_x.head()
x = np.hstack([np.ones((df_x.shape[0], 1)), df_x[["x1","x2"]].values])
y = df_y["y"].values
y = (y-min(y))/(max(y)-min(y))


def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    return g

def log_likelihood(theta,x,y):
    log_l = 0
    for i in range(len(y)):
        log_l += ( y[i] * np.log((sigmoid(np.dot(theta,x[i])))) +  (1-y[i])*np.log(1-sigmoid(np.dot(theta,x[i])))   ) 
    
    return log_l

def grad_l(theta, x, y):
    G = np.zeros(3)    
    for j in range(x.shape[1]):
        for i in range(len(y)):
            G[j] += ( (y[i] - sigmoid(np.dot(theta,x[i])))*x[i,j])
    return G

def gradient_ascent(theta,x,y,G,alpha=0.01,iterations=100):

    
    log_l_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,3))
    
    for i in range(iterations):
        G = grad_l(theta,x,y)
        for j in range(len(theta)):
            theta[j] += alpha*G[j]

        log_l_history[i] = log_likelihood(theta, x,y)
        theta_history[i] = theta
    # return the optimized theta parameters,
    # as well as two lists containing the log likelihood's and values of theta at all iterations        
    return theta, log_l_history, theta_history


target_value = -0.4250958770469834
theta_test=np.array([-2,1,2])

log_l_test  = log_likelihood(theta_test,x,y)
error_test=np.abs(log_l_test-target_value)

print("{:f}".format(error_test))

# Initialize theta0
theta0 = np.zeros(x.shape[1])

# Run Gradient Ascent method
n_iter=1000
theta_final, log_l_history, theta_history = gradient_ascent(theta0,x,y,grad_l,alpha=0.5,iterations=n_iter)
print(theta_final)
print(theta_history)
print(theta_history.shape)