from numpy import exp, round, ones, argsort, cumsum, rad2deg, deg2rad, sqrt, inf,errstate, where,sign, nan, tan, array, dot, arccos, degrees,arctan, nan, full, isnan,vstack,linspace, isfinite,arcsin, argmin,random,linalg,clip, cos, sin,pi, sqrt, arange
from scipy.optimize import minimize,newton, brentq, least_squares
from math import isclose
from pandas import DataFrame,MultiIndex
from .surface_lib import Surface
from .glass_lib import Glass
from matplotlib.pyplot import subplots,show,grid
from scipy.interpolate import interp1d
from scipy.special import erfinv
from scipy.stats import norm

class tools:
    def __init__(self): pass


    def transfer(self, y0,u0,z):
        return y0+(tan(u0)*z)


    def sag_spherical(self, y:float,R:float, z_shift:float=0):
        """
        Compute the sag of an aspherical surface at height y.

        Parameters:
        y      : Height along the y-axis
        R      : Radius of curvature (positive for convex, negative for concave)
        z_shift: The initial position of the surface at y=0

        Returns:
        z_sag : Sag (depth) of the surface at y
        """

        if R==0:
            return 0+z_shift

        if abs(y)>=abs(R):
            return nan

        elif R>0:
            return R-sqrt(R**2-y**2)+z_shift
        else:
            return R+sqrt(R**2-y**2)+z_shift


    def sag_aspherical(self,y:float, R:float, k:float, A_list:list, z_shift:float=0):
        """
        Compute the sag of an aspherical surface at height y.

        Parameters:
            y      : Height along the y-axis
            R      : Radius of curvature (positive for convex, negative for concave)
            k      : Conic constant
            A_list : List of aspheric coefficients [A4, A6, A8, ...]
            z_shift: The initial position of the surface at y=0

        Returns:
            z_sag : Sag (depth) of the surface at y
        """
        if R==0:
            return 0+z_shift
        
        if abs(y)>=abs(R):
            return nan

        base = y**2 / (R * (1 + sqrt(1 - (1 + k) * y**2 / R**2)))  # Conic sag
        higher_order = sum(A_list[i] * y**(2*(i + 2)) for i in range(len(A_list)))  # Aspheric terms
    
        z= base+ higher_order

        return z+z_shift
    

    def dzdy_spherical(self, y:float,R:float):
        """
        Compute the sag of an aspherical surface at height y.

        Parameters:
            y      : Height along the y-axis
            R      : Radius of curvature (positive for convex, negative for concave)
            
        Returns:
            z_sag : Sag (depth) of the surface at y
        """
        if R==0:
            return 0
        
        if abs(y)>=abs(R):
            return nan
        
        with errstate(invalid='ignore', divide='ignore'):
            dz = y / sqrt(R**2 - y**2)
            dz = where(R == 0, 0, dz)
            dz = where(R < 0, -dz, dz)
        return dz
    

    def dzdy_aspherical(self,y:float, R:float, k:float, A_list:list):
        """
        Compute the derivative dz/dy of the aspherical surface.

            y      : Height along the y-axis
            R      : Radius of curvature (positive for convex, negative for concave)
            k      : Conic constant
            A_list : List of aspheric coefficients [A4, A6, A8, ...]


        Returns:
            dz : Derivative of surface at point y
        """

        if R==0:
            return 0
        
        if abs(y)>=abs(R):
            return nan
        
        sqrt_term = sqrt(1 - (1 + k) * y**2 / R**2)

        if sqrt_term == 0:
            return inf  # Avoid divide by zero
        
        dz = (2 * y) / (R * (1 + sqrt_term))  # Derivative of conic sag
        dz += ((1 + k) * y**3) / (R**3 * sqrt_term * (1 + sqrt_term)**2)  # Chain rule on sqrt
        for i, a in enumerate(A_list):
            b = 2 * (i + 2)
            dz += b * a * y**(b - 1)  # Derivative of each A-term
        return dz
    

    def Angle_check(self, y0, u0, z, R, surface_type, k, A_list):
        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag
        
        y_sag= linspace(-R+1e-10, R-1e-10, 1001, endpoint=True)
        ang_u0= rad2deg(u0)
        ang= []
        for ysag in y_sag:
            if ysag==0:
                if ysag-y0==0:
                    ang.append(rad2deg(arctan(inf)))
                else:
                    ang.append(rad2deg(arctan(sign(ysag-y0)*inf)))
                # ang.append(rad2deg(arctan((ysag-y0)/SAG(ysag))))
            else:
                ang.append(rad2deg(arctan((ysag-y0)/SAG(ysag))))

        if ang_u0>min(ang) and ang_u0<max(ang):
            return True
        else:
            return False


    def _distance(self,y0, u0, z, R, surface_type, k, A_list, tol=1e-20, method='Nelder-Mead'):

        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag

        if R == 0 or isclose(u0, 0.0, abs_tol=1e-12):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan
        

        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 

        
        Methods= ['Nelder-Mead','L-BFGS-B','SLSQP']

        
        
        
        z_bnd_0= SAG(0)
        z_bnd_1= SAG(R-1e-10)

        y_bnd_0= y0+tan(u0)*z_bnd_0
        y_bnd_1= y0+tan(u0)*z_bnd_1

        bnds = [(y_bnd_0,y_bnd_1)]
        bnds=None
        if abs(R)<abs(y_bnd_0):
            return nan, nan


        epsilon = 1e-12
        def Min_Func(y_vec):
            y = y_vec[0]
            sag = SAG(y)
            line = (y - y0) / tan(u0)
            diff = sag - line
            denom = max(abs(sag), abs(line), epsilon)
            return (diff / denom) ** 2


        # Replace Line_y with a simple guess
        y_guess = y0 + tan(u0) * z

        Y_res=[]
        ERR=[]
        jac= None


        if method=='auto':
            for Method in Methods:
                
                result = minimize(Min_Func, x0=[y_guess], tol=tol, method=Method,jac=jac,options={'maxiter': 1000}, bounds=bnds)
                Y_res.append(result.x[0])
                ERR.append(result.fun)
                # print (Method,result.fun, '\n' )
            
            y_res = Y_res[argmin(ERR)]
            err = min(ERR)
            # print (err )

        elif method in Methods:
            result = minimize(Min_Func, x0=[y_guess], tol=tol, method=method, jac=jac, options={'maxiter': 1000}, bounds=bnds)
            y_res = result.x[0]
            err = result.fun
        else:
            raise ValueError(f"This Method '{method}' is not valid, available methods ['Powell', 'Nelder-Mead','CG','BFGS', 'SLSQP','L-BFGS-B']")        


        if abs(y_res) > abs(R) or err>tol:
            # print('here 4')
            return nan, nan

        return y_res, err


    def _angle2(self,y0, u0, z, R, surface_type, k, A_list, tol=1e-20, method='auto'):
        Methods= ['Nelder-Mead','CG','BFGS','L-BFGS-B','SLSQP', 'Powell']

        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag


        if R == 0 or isclose(u0, 0.0, abs_tol=1e-12):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan


        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 
            


        y_fit=linspace(-R+1e-10, R-1e-10, 2**10, endpoint=True)
        z_fit=array([SAG(yy) for yy in y_fit])
        
        ERR=abs(rad2deg(arctan((y_fit-y0)/z_fit))-rad2deg(u0))**2
        return y_fit[argmin(ERR)], min(ERR)


    def _angle(self,y0, u0, z, R, surface_type, k, A_list, tol=1e-20, method='auto'):

        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag


        if R == 0 or isclose(u0, 0.0, abs_tol=1e-12):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan
        

        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 
        
        Methods= ['Nelder-Mead','CG','BFGS','L-BFGS-B','SLSQP']
        
        epsilon = 1e-12
        def Min_Func(y_vec):
            y = y_vec[0]
            sag = SAG(y)
            line = (y - y0) / tan(u0)
            diff = arctan((y-y0)/sag)-u0
            denom = max(abs(sag), abs(line), epsilon)
            return (diff / denom) ** 2


        # Replace Line_y with a simple guess
        y_guess = y0 + tan(u0) * z

        Y_res=[]
        ERR=[]

        # if method in['SLSQP']:
        jac= None


        if method=='auto':
            for Method in Methods:
                result = minimize(Min_Func, x0=[y_guess], tol=tol, method=Method,jac=jac,options={'maxiter': 1000})
                Y_res.append(result.x[0])
                ERR.append(result.fun)

            
            y_res = Y_res[argmin(ERR)]
            err = min(ERR)


        elif method in Methods:
            result = minimize(Min_Func, x0=[y_guess], tol=tol, method=method, jac=jac, options={'maxiter': 1000})
            y_res = result.x[0]
            err = result.fun
        else:
            raise ValueError(f"This Method '{method}' is not valid, available methods ['Powell', 'Nelder-Mead','CG','BFGS', 'SLSQP','L-BFGS-B']")        


        if abs(y_res) > abs(R)or err>tol:
            return nan, nan

        return y_res, err


    def _newton(self,y0, u0, z, R, surface_type, k, A_list, tol=1e-6, max_iter=1000):
        if R == 0 or isclose(u0, 0.0, abs_tol=1e-12):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan

        # Sag and derivative functions
        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag

        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 

        def dSAG_dy(y):
            if surface_type == 'sph':
                return self.dzdy_spherical(y, R)
            elif surface_type == 'asph':
                return self.dzdy_aspherical(y, R, k, A_list)
            else:
                raise ValueError("Unknown surface_type")

        def Line_z(y):
            return (y - y0) / tan(u0)

        def dLine_z_dy(y):
            return 1 / tan(u0)

        def f(y):
            return SAG(y) - Line_z(y)

        def df(y):
            return dSAG_dy(y) - dLine_z_dy(y)

        # Initial guess: intersection assuming flat surface
        y = y0 + tan(u0) * z

        for _ in range(max_iter):
            f_val = f(y)
            df_val = df(y)

            if isnan(f_val) or isnan(df_val) or isclose(df_val, 0.0, abs_tol=1e-14):
                break  # Divergence or flat slope

            y_new = y - f_val / df_val

            if abs(y_new - y) < tol:
                if abs(y_new) > abs(R):
                    return nan, nan
                return y_new, abs(f_val)

            y = y_new

        return nan, nan  # No convergence


    def _multistart(self,y0, u0, z, R, surface_type, k, A_list, num_starts=1000, tol=1e-10):
        
        if R == 0 or isclose(u0, 0.0, abs_tol=1e-12):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan
        
        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag


        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 

        epsilon = 1e-12
        def func(y_vec):
            y = y_vec[0]
            sag = SAG(y)
            line = (y - y0) / tan(u0)
            diff = sag - line
            denom = max(abs(sag), abs(line), epsilon)
            return (diff / denom) **10

        y_bounds = (-abs(R)*1.5, abs(R)*1.5)
        results = []
        for i in range(num_starts):
            y_guess = random.uniform(*y_bounds)
            res = minimize(func, x0=[y_guess], method='Nelder-Mead', tol=tol)
            if res.success:
                results.append((res.x[0], res.fun))

        if not results:
            return nan, nan

        # Return best minimum found
        y_res, err = min(results, key=lambda x: x[1])
        if abs(y_res) > abs(R) or err>tol:
            return nan, nan
        
        return y_res, err


    def _trust_constr(self,y0, u0, z, R, surface_type, k, A_list, tol=1e-10):
        
        if R == 0 or isclose(u0, 0.0, abs_tol=1e-12):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan
        
        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':

                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag

        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 



        epsilon = 1e-12
        def func(y_vec):
            y = y_vec[0]
            sag = SAG(y)
            line = (y - y0) / tan(u0)
            diff = sag - line
            denom = max(abs(sag), abs(line), epsilon)
            return (diff / denom) ** 2

        y_guess = y0 + tan(u0) * z
        bounds = [(-abs(R)*2, abs(R)*2)]  # wide bounds to avoid surface domain issues
        result = minimize(func, x0=[y_guess], method='trust-constr', bounds=bounds, tol=tol)

        y_res = result.x[0]
        err = result.fun
        if abs(y_res) > abs(R) or err>tol:
            return nan, nan
        return y_res, err


    def _trust_constr_with_jac(self,y0, u0, z, R, surface_type, k, A_list, tol=1e-10):

        if R == 0 or isclose(u0, 0.0, abs_tol=1e-20):
            return y0 + tan(u0) * z, 0

        if isnan(y0) or isnan(u0):
            return nan, nan

        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':
                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag


        if not self.Angle_check(y0, u0, z, R,surface_type, k, A_list):
            return nan, nan 



        epsilon = 1e-12
        def func(y_vec):
            y = y_vec[0]
            sag = SAG(y)
            line = (y - y0) / tan(u0)
            diff = sag - line
            denom = max(abs(sag), abs(line), epsilon)
            return (diff / denom) ** 2


        def jac(y_vec):
            y = y_vec[0]
            sag = SAG(y)
            # derivative sag'
            if surface_type == 'sph':
                dsag_dy = self.dzdy_spherical(y, R)
            elif surface_type == 'asph':
                dsag_dy = self.dzdy_aspherical(y, R, k, A_list)
            else:
                raise ValueError("Unknown surface_type")
            
            dline_dy = 1 / tan(u0)
            grad = 2 * (sag - (y - y0)/tan(u0)) * (dsag_dy - dline_dy)
            return array([grad])

        y_guess = y0 + tan(u0) * z
        bounds = [(-abs(R)*2, abs(R)*2)]
        result = minimize(func, x0=[y_guess], method='trust-constr', jac=jac, bounds=bounds, tol=tol)
        y_res = result.x[0]
        err = result.fun
        if abs(y_res) > abs(R) or err>tol:
            return nan, nan
        return y_res, err


    def intersection(self,y0, u0, z, R, surface_type, k=0, A_list=[], intersection_method='auto', tol=1e-10):
        """
        Compute the intersection between ray and a surface.

        Parameters:
            y0                  : Initial Height along the y-axis of the ray
            u0                  : The ray propagation angle against the optical axis
            z                   : The horizantal displacement between the ray (y0,u0) and the surface on the optical axis i.e. at height of 0 for the srufaces
            R                   : Radius of curvature (positive for convex, negative for concave)
            surface_type        : The Type of Surface 'asph' or 'sph'
            k                   : Conic constant
            A_list              : List of aspheric coefficients [A4, A6, A8, ...]
            intersection_method : The method to perform the intersection finding Auto Selection: 'auto' (default), distance: 'd', angle: 'a', angle2: 'a2', newton:'n', multi_start:'m', trust_constr: 'tc', trust_constr_with_jac: 'tcj'
            tol                 : The acceptance tolerance default 1e-10

        Returns:
            y_int               : The height of the ray at which intersects with the surface
            err                 : The error of the results
        """
        if R==0:
            yres= self.transfer(y0, u0, z)
            return yres, 0

        if intersection_method.lower() =='auto':
            Methods=[self._distance, self._angle,self._newton, self._multistart, self._trust_constr, self._trust_constr_with_jac, self._angle2]
            _Methods= ['Distance', 'Angle', 'Newton', 'Multistart', 'Trust', 'Trust_jac', 'Angle2']
            ErrRes=[]
            YRes=[]

            for method in Methods:
                yres, errres=method(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)
                if not isnan(yres):
                    ErrRes.append(errres)
                    YRes.append(yres)
            # print(YRes,ErrRes)
            if len(YRes)==0:
                return nan, nan
            # print(_Methods[argmin(ErrRes)])
            return YRes[argmin(ErrRes)], min(ErrRes)
        
        elif intersection_method.lower() in ['distance', 'dist', 'd']:
            yres, errres=self._distance(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)


        
        elif intersection_method.lower() in ['angle', 'ang', 'a']:
            yres, errres=self._angle(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)

        elif intersection_method.lower() in ['angle2', 'ang2', 'a2']:
            yres, errres=self._angle2(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)



        elif intersection_method.lower() in ['newton', 'new', 'n']:
            yres, errres=self._newton(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)

       
        elif intersection_method.lower() in ['multi_start', 'multi', 'multistart', 'start', 'ms', 'm_s', 'm-s', 'm','s']:
            yres, errres=self._multistart(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)

      
        elif intersection_method.lower() in ['trust_constr', 'trust', 'constr', 'trustconstr', 't', 'tc', 't-c', 't_c']:
            yres, errres=self._trust_constr(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)

      
        elif intersection_method.lower() in ['trust_constr_with_jac','trust_constr_jac', 'trustconstrjac', 'tcj', 't-c-j', 't_c_j', 'cj']:
            yres, errres=self._trust_constr_with_jac(y0, u0, z, R, surface_type, k=k, A_list=A_list, tol=tol)
        
        else:
            raise ValueError(f"This Method '{intersection_method}' is not valid, available methods are:\nAuto Selection: 'auto' (default)\ndistance: 'd'\nangle: 'a'\nnewton:'n'\nmulti_start:'m'\ntrust_constr: 'tc'\ntrust_constr_with_jac: 'tcj'")        
        
        # print(yres, errres)
        if isnan(yres):
            return nan, nan
        return yres, errres


    def compute_u_incident2(self, y_int, u0, R, surface_type, k=0, A_list=[]):
        """
        Compute the angle between incoming ray and the normal to the surface (angle of incidence).

        Parameters:
        - y_int: y-coordinate of intersection
        - u0: angle of incoming ray in radians
        - R: surface radius
        - surface_type: 'sph' or 'asph'
        - k: conic constant (used if aspherical)
        - A_list: asphere coefficients (used if aspherical)

        Returns:
        - u_incident: angle between ray and surface normal [radians]
        """

        # Get dz/dy (surface slope) at y_int
        if surface_type == 'sph':
            slope = self.dzdy_spherical(y_int, R)
        elif surface_type == 'asph':
            slope = self.dzdy_aspherical(y_int, R, k, A_list)
        else:
            raise ValueError("Invalid surface type. Use 'sph' or 'asph'.")

        if isnan(slope):
            return nan

        # Surface normal vector
        n = array([-slope, 1])
        n = n / linalg.norm(n)

        # Ray direction vector from u0
        v = array([cos(u0), sin(u0)])
        v = v / linalg.norm(v)

        # Angle between ray and normal
        dot_prod = clip(dot(v, n), -1.0, 1.0)  # clip for numerical stability
        angle = arccos(dot_prod)

        return angle, slope


    def compute_u_incident2(self, y_int, u0, R, surface_type, k=0, A_list=[]):
            """
            Compute the angle between incoming ray and the normal to the surface (angle of incidence).

            Parameters:
            - y_int: y-coordinate of intersection
            - u0: angle of incoming ray in radians
            - R: surface radius
            - surface_type: 'sph' or 'asph'
            - k: conic constant (used if aspherical)
            - A_list: asphere coefficients (used if aspherical)

            Returns:
            - u_incident: angle between ray and surface normal [radians]
            """

            # Get dz/dy (surface slope) at y_int
            if R==0:

                return u0, inf
            

            if surface_type == 'sph':
                dzdy = self.dzdy_spherical(y_int, R)
            elif surface_type == 'asph':
                dzdy = self.dzdy_aspherical(y_int, R, k, A_list)
            else:
                raise ValueError("Invalid surface type. Use 'sph' or 'asph'.")

            if isnan(dzdy):

                return nan, nan
            if isclose(dzdy, 0.0, abs_tol=1e-50):
                
                slope= inf
            else:

                slope=1/dzdy
            # print(u0, slope)
            if sign(slope)==1:
                return deg2rad(90)+u0-arctan(slope), slope
            else:
                return (deg2rad(90)+u0-arctan(slope)-deg2rad(180)), slope


    def Snell(self, u1, n1, n2):
        """
        Apply Snell's Law to compute the refraction angle at an interface.

        Parameters:
        - u1: Incident angle in radians (measured from the surface normal)
        - n1: Refractive index of the initial medium
        - n2: Refractive index of the second medium

        Returns:
        - u2: Refracted angle in radians (measured from the surface normal),
            or nan if input is invalid or total internal reflection occurs.

        Notes:
        - If u1 is NaN or n1*sin(u1)/n2 > 1, the function returns nan.
        """
        # print(f'u1:{u1}, n1:{n1}, n2:{n2}')
        sin_u2 = n1 * sin(u1) / n2
        if isnan(u1) or sin_u2>1 or sin_u2<-1:
            return nan  # Invalid input

        # print(f'sin_u2: {sin_u2}, u2:{arcsin(sin_u2)}')
        # print(sin_u2)

        return arcsin(sin_u2)
    

    def _solver(self,y0:float,u0:float,surf_crnt,surf_nxt,z:float=None, intersection_method:str='distance', tol:float= 1e-10):

        """
        y0                  : input Ray height (mm) at current surface
        u0                  : Ray Propagation angle (rad) with respect to the optical axis
        surf_crnt           : The properties dataframe of the current surface index
        surf_nxt            : The properties dataframe of the next surface index
        z                   : The displacement (mm) along the optical axis (the distance between current surface and next surface at height y=0), if None then z= surf_crnt.thickness
        intersection_method : The method to perform the intersection finding Auto Selection: 'auto' (default), distance: 'd', angle: 'a', newton:'n', multi_start:'m', trust_constr: 'tc', trust_constr_with_jac: 'tcj'
        tol                 : The acceptance tolerance default 1e-10
        """


        # print(surf_crnt, surf_nxt, '\n\n\n\n\n')
        if z==None:
            z= surf_crnt.thickness
        
        mat_crnt= surf_crnt.material

        R= surf_nxt.radius
        k= surf_nxt.conic
        A_list= surf_nxt.A_coefficient
        aprature= surf_nxt.aperture
        surface_type= surf_nxt.type
        mat_nxt= surf_nxt.material

        if surf_nxt.loc['surface_name']=='ims':

            return self.transfer(y0,u0, z+surf_nxt.thickness), u0,z+surf_nxt.thickness, 0
        
        def SAG(y):
            if surface_type == 'sph':
                sag = self.sag_spherical(y, R, z)
                if isnan(sag):
                    sag = self.sag_spherical(sign(y) * (abs(R) - 1e-10), R, z)
            elif surface_type == 'asph':
                sag = self.sag_aspherical(y, R, k, A_list, z)
                if isnan(sag):
                    sag = self.sag_aspherical(sign(y) * (abs(R) - 1e-10), R, k, A_list, z)
            else:
                raise ValueError("Unknown surface_type")
            
            return sag

        y1, err= self.intersection(y0, u0, z, R, surface_type, k, A_list, intersection_method=intersection_method, tol=tol)
        
        z_int= SAG(y1)
        

        n1= self.get_Ref_Index(self.wl, mat_crnt, None)
        n2= self.get_Ref_Index(self.wl, mat_nxt, None)

        
        
        u0_incid, slope= self.compute_u_incident2(y1, u0, R,surface_type,k,A_list)

        ur= self.Snell(u0_incid, n1,n2)
        u1= arctan(slope)+ur-deg2rad(90)
        if sign(slope)==-1:
            u1= arctan(slope)+ur-deg2rad(90)+deg2rad(180)

        return y1, u1, z_int, err


    # def gaussian_fiber(self,n_rays:int, MFD_um=None, NA=None, n_core=None, n_clad=None, r_core_NA=None, fiber_type="SMF"):
    #     """
    #     Generates deterministic 1D fiber rays.
    #     Units:
    #         - n_rays    : Number or rays
    #         - MFD       : Mode field diameter in µm (Opition 1)
    #         - NA        : Numerical aperture of the fiber (option 2), to generate rays via NA method r_core_NA, is needed to be passed too, you can pass the n_core and n_clad to calculate the NA (keep NA=None)
    #         - n_core    : Refractive index of the core
    #         - n_clad    : Refractive index of the clading
    #         - r_core_NA : Core radius in µm
    #         - fiber_type: type of the fiber, Single-Mode (SMF) or Multi-Mode (MMF)

    #     Returns:
    #         h_obj: Ray height in mm 
    #         u_rad: Ray angle in rad
    #     """
    #     MFD=MFD_um
    #     H_obj = []
    #     U_rad = []

    #     # Convert wavelength to mm for calculation
    #     wavelength_mm = self.wl * 1e-3

    #     if fiber_type.lower() == "smf":
    #         if MFD is not None:
    #             w0 = MFD* 1e-3 / 2.0  # mode radius in mm
    #         else:
    #             if r_core_NA is None:
    #                 raise ValueError("Core radius 'r_core_NA' must be provided if MFD not given for SMF.")
    #             if NA is None:
    #                 if n_core is not None and n_clad is not None:
    #                     NA = sqrt(n_core**2 - n_clad**2)
    #                 else:
    #                     raise ValueError("Either NA or (n_core and n_clad) must be provided for SMF if MFD not given.")
    #             V = 2 * pi * r_core_NA*1e-3 / wavelength_mm * NA
    #             w0 = r_core_NA*1e-3 * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)

    #         # Deterministic Gaussian sampling
    #         u = (arange(n_rays) + 0.5) / n_rays
    #         h_obj = w0 * sqrt(2) * erfinv(2*u - 1)       # mm
    #         theta_div = wavelength_mm / (pi * w0)
    #         u_rad = theta_div * sqrt(2) * erfinv(2*u - 1)  # rad

    #     elif fiber_type.lower() == "mmf":
    #         if r_core_NA is None:
    #             raise ValueError("Core radius 'r_core_NA' must be provided for MMF.")
    #         if NA is None:
    #             if n_core is not None and n_clad is not None:
    #                 NA = sqrt(n_core**2 - n_clad**2)
    #             else:
    #                 raise ValueError("Either NA or (n_core and n_clad) must be provided for MMF.")
    #         h_obj = linspace(-r_core_NA, r_core_NA, n_rays)*1e-3  # mm
    #         theta_max = arcsin(NA)
    #         u_rad = linspace(-theta_max, theta_max, n_rays)  # rad

    #     for i in range(n_rays):
    #         H_obj.append(h_obj[i])
    #         U_rad.append(u_rad[i])

    #     return H_obj, U_rad



    # def gaussian_beam_rays(self,n_rays, w0_mm, z_mm=0.0, truncate_sigma=1.0):
    #     """
    #     Generate deterministic Gaussian beam rays at any longitudinal position z
    #     with truncated heights and divergence angle for a specified beam waist
        
    #     Parameters:
    #         - n_rays         : Number of rays
    #         - w0_mm             : Beam waist radius [mm]
    #         - z_mm              : Longitudinal position [mm] (z=0 is waist)
    #         - truncate_sigma : Truncation of Gaussian in sigma units (e.g., 1 sigma)
                
    #     Returns:
    #         h_obj: Ray height in mm 
    #         u_rad: Ray angle in rad
    #     """
    #     w0= w0_mm
    #     z= z_mm
    #     # Convert wavelength to mm
    #     lam = self.wl * 1e-3
        
    #     # Rayleigh range
    #     zR = pi * w0**2 / lam
        
    #     # Beam radius at z
    #     w_z = w0 * sqrt(1 + (z/zR)**2)
        
    #     # Wavefront radius of curvature
    #     Rz = inf if z == 0 else z * (1 + (zR/z)**2)
        
    #     # Gaussian standard deviations
    #     sigma_h = w_z / sqrt(2)
    #     sigma_theta = (lam/(pi*w0)) / sqrt(2)  # small divergence at waist
        
    #     # Truncate Gaussian in height
    #     cdf_min = norm.cdf(-truncate_sigma*sigma_h, scale=sigma_h)
    #     cdf_max = norm.cdf( truncate_sigma*sigma_h, scale=sigma_h)
        
    #     # Deterministic sampling in truncated CDF
    #     u = linspace(cdf_min, cdf_max, n_rays)
        
    #     # Heights: truncated Gaussian
    #     h_obj = sigma_h * sqrt(2) * erfinv(2*u - 1)
        
    #     # Angles: small divergence Gaussian
    #     u_rad = sigma_theta * sqrt(2) * erfinv(2*u - 1)
        
    #     # Optionally: add wavefront curvature
    #     if isfinite(Rz):
    #         u_rad += h_obj / Rz  # correlates angle with wavefront curvature
        
    #     return h_obj, u_rad


    def gaussian_fiber(self, n_rays, MFD_um=None, NA=None, r_core_um=None, method="deterministic", seed=None):
        """
        Generate SMF Gaussian rays using either MFD or NA+core radius.

        Parameters
        ----------
        n_rays : int
            Number of rays
        wavelength_um : float
            Wavelength [µm]
        MFD_um : float, optional
            Mode Field Diameter [µm]. If given, NA and r_core are ignored.
        NA : float, optional
            Numerical aperture of the fiber. Must provide r_core_um if MFD_um not given.
        r_core_um : float, optional
            Fiber core radius [µm] (needed if NA is used)
        method : str
            "deterministic" or "random"
        seed : int or None
            Seed for random sampling (only used if method="random")

        Returns
        -------
        h_obj : ndarray
            Ray heights [mm]
        u_rad : ndarray
            Ray angles [rad]
        weights : ndarray
            Gaussian weights normalized to sum = 1
        """
        wavelength_mm = self.wl * 1e-3  # convert to mm

        # Determine mode radius w0
        if MFD_um is not None:
            w0 = MFD_um * 1e-3 / 2.0
        elif NA is not None and r_core_um is not None:
            r_core_mm = r_core_um * 1e-3
            V = 2 * pi * r_core_mm / wavelength_mm * NA
            w0 = r_core_mm * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)
        else:
            raise ValueError("Either MFD_um or (NA and r_core_um) must be provided.")

        # Gaussian divergence half-angle
        theta_div = wavelength_mm / (pi * w0)

        # Generate rays
        if method.lower() == "deterministic":
            u = (arange(n_rays) + 0.5) / n_rays
            h_obj = w0 * sqrt(2) * erfinv(2*u - 1)
            u_rad = theta_div * sqrt(2) * erfinv(2*u - 1)
        elif method.lower() == "random":
            rng = random.default_rng(seed)
            h_obj = rng.normal(0, w0/sqrt(2), n_rays)
            u_rad = rng.normal(0, theta_div/sqrt(2), n_rays)
        else:
            raise ValueError("method must be 'deterministic' or 'random'")

        # Gaussian weights
        weights = exp(-2 * (h_obj / w0)**2)
        weights /= sum(weights)

        return h_obj, u_rad, weights


    def gaussian_beam_rays(self, n_rays, w0_mm, z_mm=0.0, truncate_sigma=3.0, method="deterministic", seed=None):
        """
        Generate Gaussian beam rays at any longitudinal position z with optional truncation.
        Supports deterministic or random sampling and returns Gaussian weights.

        Parameters
        ----------
        n_rays : int
            Number of rays
        w0_mm : float
            Beam waist radius [mm]
        wavelength_um : float
            Wavelength [µm]
        z_mm : float
            Longitudinal position [mm] (z=0 is waist)
        truncate_sigma : float
            Truncation in sigma units

            •	truncate_sigma = 1 → ±1σ   → ~68% of power
            •	truncate_sigma = √2 → ±√2σ → ~86.5% of power
	        •	truncate_sigma = 2 → ±2σ   → ~95% of power
	        •	truncate_sigma = 3 → ±3σ   → ~99.7% of power

        method : str
            "deterministic" or "random"
        seed : int or None
            Seed for random sampling (only used if method="random")

        Returns
        -------
        h_obj : ndarray
            Ray heights [mm]
        u_rad : ndarray
            Ray angles [rad]
        weights : ndarray
            Gaussian weights normalized to sum = 1
        """
        lam_mm = self.wl * 1e-3

        # Rayleigh range
        zR = pi * w0_mm**2 / lam_mm

        # Beam radius at z
        w_z = w0_mm * sqrt(1 + (z_mm/zR)**2)

        # Wavefront radius of curvature
        Rz = inf if z_mm == 0 else z_mm * (1 + (zR/z_mm)**2)

        # Standard deviations
        sigma_h = w_z / sqrt(2)
        sigma_theta = (lam_mm/(pi*w0_mm)) / sqrt(2)

        # Generate heights and angles
        if method.lower() == "deterministic":
            # truncated CDF sampling
            cdf_min = norm.cdf(-truncate_sigma*sigma_h, scale=sigma_h)
            cdf_max = norm.cdf( truncate_sigma*sigma_h, scale=sigma_h)
            u = linspace(cdf_min, cdf_max, n_rays)
            h_obj = sigma_h * sqrt(2) * erfinv(2*u - 1)
            u_rad = sigma_theta * sqrt(2) * erfinv(2*u - 1)
        elif method.lower() == "random":
            rng = random.default_rng(seed)
            h_obj = rng.normal(0, sigma_h, n_rays)
            # truncate if requested
            if truncate_sigma > 0:
                mask = abs(h_obj) > truncate_sigma * sigma_h
                while any(mask):
                    h_obj[mask] = rng.normal(0, sigma_h, sum(mask))
                    mask = abs(h_obj) > truncate_sigma * sigma_h
            u_rad = rng.normal(0, sigma_theta, n_rays)
        else:
            raise ValueError("method must be 'deterministic' or 'random'")

        # Apply wavefront curvature
        if isfinite(Rz):
            u_rad += h_obj / Rz

        # Gaussian weights based on height
        weights = exp(-2 * (h_obj/w_z)**2)
        weights /= sum(weights)

        return h_obj, u_rad, weights



