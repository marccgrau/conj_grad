import warnings
import numpy as np
import tensorflow as tf

#TODO: quite probably the model should be passed along as function and the gradients \
#      could be computed potentially computed within and do not have to be passed as a callable \
#      But how is this gonna work without the loss being passed as a callable?

#TODO: Check where one has to put tf.function decorators

@tf.function
def wolfe_line_search(f, f_grad, xk, dk, c1=1e-4, c2=0.1, amax=1.0, maxiter=10):
    """
    Find alpha that satisfies the strong Wolfe conditions.
    
    Parameters
    ----------
    f: callable f(x)
        Objective function.
    f_grad: callable f'(x)
        Gradient of the objective function.
    xk: tf.Tensor
        Starting point.
    dk: tf.Tensor
        Search direction.
    c1: float, optional
        Parameter for the sufficient decrease condition. (Armijo condition rule)
    c2: float, optional
        Parameter for the curvature condition. (Wolfe condition rule)
    amax: float, optional
        Maximum step size.
    maxiter: int, optional
        Maximum number of iterations.
    
    Returns
    -------
    alpha: float
        Alpha for which ``x_new = x0 + alpha * pk`` 
        or None if the line search did not converge
    phi: float or None
        New function value ``f(x_new) = f(x0 + alpha * pk)``
        or None if the line search did not converge.
        
        """
    
    
    #TODO: Make sure all tf.Tensors as inputs
    @tf.function
    def phi(alpha):
        return f(xk + alpha * dk)
    
    #TODO: Either tape.gradient as input for f_grad or compute it here
    @tf.function
    def derphi(alpha):
        return tf.tensordot(f_grad(xk + alpha * dk), dk)
    
    alpha_star, phi_star, derphi_star = wolfe_line_search_2(phi, derphi, c1, c2, amax, maxiter)
    
    if derphi_star is None:
        warnings.warn('Line search did not converge')
    
    return alpha_star, phi_star

@tf.function
def wolfe_line_search_2(phi, derphi, c1=1e-4, c2=0.9, amax=None, maxiter=10):
    """
    Find alpha that satisfies strong Wolfe conditions.
    alpha > 0 is assumed to be a descent direction. #NOTE: Not always the case for Polak-Ribiere
    Parameters
    ----------
    phi: callable phi(alpha)
        Objective scalar function.
    derphi: callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    c1: float, optional
        Parameter for Armijo condition rule.
    c2: float, optional
        Parameter for curvature condition rule.
    amax: float, optional
        Maximum step size.
    maxiter: int, optional
        Maximum number of iterations.
    Returns
    -------
    alpha_star: float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star: float
        phi at alpha_star.
    derphi_star: float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.
    """
    phi0, derphi0 =
    
    alpha0 = 0.
    alpha1 = 1.
    
    if amax is not None:
        alpha1 = tf.math.minimum(alpha1, amax)
    
    phi_a1 = phi(alpha1)
    
    phi_a0 = phi0
    derphi_a0 = derphi0
    
    for i in tf.range(maxiter):
        if tf.math.logical_or(tf.math.equal(alpha1, 0), tf.math.logical_and(amax is not None, tf.math.equal(alpha0, amax))):
            alpha_star = None
            phi_star = phi0
            derphi_star = None
            
            if tf.math.equal(alpha1, 0):
                warnings.warn('Rounding errors preventing line search from converging')
            else:
                warnings.warn(f'Line search could not find solution less than or equal to {amax}')
            break
        
        # test first condition of line search as illustrated in paper
        if tf.math.logical_or(tf.math.greater(phi_a1, phi0 + c1 * alpha1 * derphi0), tf.math.logical_and(tf.math.greater_equal(phi_a1, phi_a0), tf.math.greater(i, 1))):
            alpha_star, phi_star, derphi_star = _zoom(alpha0,
                                                    alpha1,
                                                    phi_a0,
                                                    phi_a1, 
                                                    derphi_a0, 
                                                    phi, 
                                                    derphi,
                                                    phi0, 
                                                    derphi0, 
                                                    c1, 
                                                    c2,
                                                    )
            break
        
        # test second condition of line search
        derphi_a1 = derphi(alpha1) # necessary for testing second condition
        if tf.math.lower_equal(tf.math.abs(derphi_a1), -c2 * derphi0):
            alpha_star = alpha1 # suitable alpha found set to star and return
            phi_star = phi(alpha1) 
            derphi_star = derphi_a1
            break
        
        # check for third condition
        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = _zoom(alpha1,
                                                    alpha0,
                                                    phi_a1,
                                                    phi_a0,
                                                    derphi_a1,
                                                    phi,
                                                    derphi,
                                                    phi0,
                                                    derphi0,
                                                    c1,
                                                    c2,
                                                    )
            break
        
        # extrapolation step of alpha_i as no conditions are met
        # simple procedure to mulitply alpha by 2
        alpha2 = 2*alpha1
        # check if we don't overshoot amax
        if amax is not None:
            alpha2 = min(alpha2,amax)
        
        # update values for next iteration to check conditions
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    
    # if no break occurs, then we have not found a suitable alpha after maxiter
    else:
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warnings.warn('Line search did not converge')
    
    return alpha_star, phi_star, derphi_star

def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found, return None.
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
        phi, derphi, phi0, derphi0, c1, c2):
    """
    Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star

def nonlinear_cg(f, f_grad, init, method='PR', c1=1e-4, c2=0.1, amax=None, tol=1e-7, max_iter=1000):
    """
    Non Linear Conjugate Gradient Method for optimization problem.
    Given a starting point x ∈ ℝⁿ.
    repeat
        1. Calculate step length alpha using Wolfe Line Search.
        2. Update x_new = x + alpha * p.
        3. Calculate beta using one of available methods.
        4. Update p = -f_grad(x_new) + beta * p
    until stopping criterion is satisfied.
    
    Parameters
    --------------------
        f        : function to optimize
        f_grad   : first derivative of f
        init     : initial value of x, can be set to be any numpy vector,
        method   : method to calculate beta, can be one of the followings: FR, PR, HS, DY, HZ.
        c1       : Armijo constant
        c2       : Wolfe constant
        amax     : maximum step size
        tol      : tolerance of the difference of the gradient norm to zero
        max_iter : maximum number of iterations
        
    Returns
    --------------------
        curve_x  : x in the learning path
        curve_y  : f(x) in the learning path
    """
    
    # init values
    x = init
    y = f(x) 
    gfk = f_grad(x) # gradient of f at x
    p = -gfk # initial search direction
    gfk_norm = np.linalg.norm(gfk)
    
    # result tabulation
    num_iter = 0
    curve_x = [x]
    curve_y = [y]
    print(f'Initial condition: y = {y}, x = {x}')
    
    # begin iteration
    while gfk_norm > tol and num_iter < max_iter:
        # search for step size alpha
        alpha , y_new = wolfe_line_search(f, f_grad, x, p, c1=c1, c2=c2, amax=amax)
        
        # update iterate x
        x_new = x + alpha * p
        gf_new = f_grad(x_new) # gradient of f at x_new
        
        # calculate beta
        if method == 'FR':
            beta = np.dot(gf_new, gf_new) / np.dot(gfk, gfk)
        elif method == 'PR':
            y_hat = gf_new - gfk
            beta = np.dot(gf_new, y_hat) / np.dot(gfk, gfk)
        elif method == 'HS':
            y_hat = gf_new - gfk
            beta = np.dot(y_hat, gf_new) / np.dot(y_hat, p)
        elif method == 'DY':
            y_hat = gf_new - gfk
            beta = np.dot(gf_new, gf_new) / np.dot(y_hat, p)
        elif method == 'HZ':
            y_hat = gf_new - gfk
            beta = np.dot(y_hat, gf_new) / np.dot(y_hat, p)
            beta = beta - 2 * np.dot(y_hat, y_hat) * np.dot(p, gf_new) / (np.dot(y_hat, p) ** 2)
        else:
            raise ValueError('Invalid method! Try one of the followings: FR, PR, HS, DY, HZ.')
            
        # update everything
        error = y - y_new
        x = x_new
        y = y_new
        gfk = gf_new
        p = -gfk + beta * p
        gfk_norm = np.linalg.norm(gfk)
        
        # tabulate results
        num_iter += 1
        curve_x.append(x)
        curve_y.append(y)
        print(f'Iteration: {num_iter}, y = {y}, x = {x}, gradient = {gfk_norm}')
    
    # print results
    if num_iter == max_iter:
        print('Maximum number of iterations reached! Gradient does not converge')
    else:
        print(f'Solution: y = {y}, x = {x}')
    
    return np.array(curve_x), np.array(curve_y)
