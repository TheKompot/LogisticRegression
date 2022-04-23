import numpy as np
from typing import Callable

def DFP(x0:np.array,f:Callable[[np.array],float], df:Callable[[np.array],np.array],
        optimal_step:bool, backtrack_pam:dict = None, H:np.array=None,
        max_iter:int=10000, eps:float=1e-3)->np.array:
    ''' 
    Uses DFP optimalization method to calculate minimum of function "f"

    Parameters:
    -----------
    x0 : np.array
        starting point
    f : Callable[[np.array],float]
        function to be optimized
    df : Callable[[np.array],np.array]
        derivation of function "f"
    optimal_step : bool:
        if True, function will use bisection method to calculate the lenght of step 
        if False, function will use backtracking method to calculate the lenght of step
    backtrack_pam : dict, optional
        aditional parameters when using backtracking for getting step size, must have:
            alpha:float - no fucking clue what this is, default 0.25
            delta:float - fraction by which the step will be decreased, default 0.7
    H : np.array, optional
        matrix of the second derivation at point x0, default identity matrix
    max_iter : int, optional
        maximal number of iteration, default 10 000
    eps : float, optional
        minimal norm of the gradient vector, before algorithm stops, default 0.001
    
    Returns
    --------
    x_opt : np.array
        point where the minimum was calculated
    '''

    x = x0
    g = df(x) #gradient

    if H is None:
        H = np.identity(len(x))

    for i in range(max_iter):
        s = -H@g  #direction
        if optimal_step:
            c = bisection(lambda phi: df(x + phi*s)@s)
        else:
            if backtrack_pam is not None:
                c = backtracking(f, x, s, g, delta=backtrack_pam['delta'], alpha=backtrack_pam['alpha'])
            else:
                c = backtracking(f, x, s, g)
        new_x = x + c*s # new point
        new_g = df(new_x) #gradient

        if np.linalg.norm(new_g) < eps: # norm of gradient smaller then eps -> end
            x = new_x
            break

        y = new_g - g
        p = new_x - x

        deltaH = (1/np.inner(p,y))*np.outer(p,p) - (1/(y@H@y))*(H@np.outer(y,y)@H) #DFP formula

        H = H + deltaH
        x = new_x
        g = new_g
        
    return x

def bisection(df,a0=0,b0=1,eps=1e-3,n=1000):
    '''Searching minimum between "a0" and "b0" of a 1d function, which is antiderivation of "df"'''
    for _ in range(n):
        if b0-a0 < eps:
            break
        c = (b0+a0)/2
        dc = df(c)

        if df(c) < 0:
            a0 = c
        else:
            b0 = c
    return c

def backtracking(f, x, s, grad, delta=0.7, alpha=0.25,c=1):
    '''Searching sub-optimal step in the direction of "s" from "x" in function "f" '''
    fx = f(x)
    s_times_grad = np.inner(s, grad)
    while f(x + c*s) > fx + alpha * c * s_times_grad:
        c = c*delta
    return c