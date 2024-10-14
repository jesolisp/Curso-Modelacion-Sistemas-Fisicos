import numpy as np

def get_kr(sys, K):
    """
     This function finds the scale factor kr which will
     eliminate the steady-state error to a step reference.
     
                               /---------\
           ref       +     u  | .       |
          ---> kr --->() ---->| X=Ax+Bu |--> y=Cx ---> y
                     -|       \---------/
                      |             | 
                      |<---- K <----|

    Parameters
    ----------
    sys : StateSpace of a linear system
        SISO continuous-time using ctrl.ss().
    K : Vector of dimension [1xn]
        Feedback vector K.

    Returns
    -------
    Array of float64
        Scale factor kr.

    """
    s = sys.A.shape[0]
    Z = np.hstack((np.zeros((1,s)), [[1]]))
    N = np.dot(np.linalg.inv(np.vstack((np.hstack((sys.A, sys.B)), np.hstack((sys.C, sys.D))))), Z.T)
    Nx = N[0:s]
    Nu = N[s]
    return Nu + np.dot(K,Nx)