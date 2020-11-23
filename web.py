def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    return g

def log_likelihood(theta,x,y):
    log_l = 0
    for i in range(len(y)):
        log_l += ( y[i] * np.log((sigmoid(np.dot(theta,x[i])))) + (1-y[i])*np.log(1-sigmoid(np.dot(theta,x[i])))) 
    
    return log_l

def grad_l(theta, x, y):
    G = np.zeros(3)    
    for j in range(x.shape[1]):
        for i in range(len(y)):
            G[j] +=( (y[i] - sigmoid(np.dot(theta,x[i])))*x[i,j])
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