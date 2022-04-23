import numpy as np

def DFP(x0:np.array,f:function, df:function, optimal_step:bool, H:np.array=None, max_iter:int=1000, eps:float=1e-3)->np.array:
    ''' 
    Uses DFP optimalization method to calculate minimum of function "f"

    Parameters:
    -----------
    x0 : np.array
        starting point
    f : functiom
        function to optimize
    df : function
        derivation of function "f"
    optimal_step : bool:
        if True, function will use bisection method to calculate the lenght of step 
        if False, function will use backtracking method to calculate the lenght of step
    H : np.array, optional
        matrix of the second derivation at point x0, default identity matrix
    max_iter : int, optional
        maximal number of iteration, default 1000
    eps : float, optional
        minimal change of function value f, before algorithm stops, default 0.001
    
    Returns
    --------
    x_opt : np.array
        point where the minimum of function "f" was calculated
    '''

    x = x0
    g = df(x) #gradient

    for i in range(max_iter):
        s = -H@g  #direction
        if optimal_step:
            c = bisection()
        else:
            c = backtracking()
        new_x = x + c*s # new point
        new_g = df(x2) #gradient

        if np.linalg.norm(new_g) < eps: # norm of gradient smaller then eps -> end
            break

        y = new_g - g
        p = new_x - x

        deltaH = (1/np.inner(p,y))*np.outer(p,p) - (1/(y@H@y))*(H@np.outer(y,y)@H) #DFP formula

        H = H + deltaH
        x = new_x
        g = new_g
        
    return x

def bisection():
    return 0

def backtracking():
    return 0