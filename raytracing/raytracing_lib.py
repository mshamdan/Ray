from numpy import arctan2, pi, argwhere, mean, round, ones, argsort, cumsum, rad2deg, deg2rad, sqrt, inf,errstate, where,sign, nan, tan, array, dot, arccos, degrees,arctan, nan, full, isnan,vstack,linspace, isfinite,arcsin, argmin,random,linalg,clip, cos, sin
from scipy.optimize import minimize_scalar, minimize,newton, brentq, least_squares
from math import isclose
from pandas import DataFrame,MultiIndex
from .surface_lib import Surface
from .glass_lib import Glass
from .toolbox import tools
from matplotlib.pyplot import subplots,show,grid
from scipy.interpolate import interp1d
from warnings import warn


class RayTracing(Surface,Glass,tools):
    def __init__(self, wl_um:float=0.58756, beam_radius:float=1, object_height:float=0,Ent_radius:float=0, Ent_material:str='air', Obj_2_Ent_tickness:float= 1e20, Ent_thickness:float=0, Ent_aperture:float=1e20, Ent_surface_type:str='sph', Ent_conic:float=0,Ent_A_list:list=[]):
        Surface.__init__(self, Ent_radius, Ent_material, Obj_2_Ent_tickness, Ent_thickness, Ent_aperture, Ent_surface_type, Ent_conic,Ent_A_list)
        Glass.__init__(self)
        tools.__init__(self)
        
        self.wl= wl_um
        self.beam_radius= beam_radius # at ENT surface
        self.h_obj= object_height # at the origin point source position if we asumed the point source (0,0+object_height,0)      
        self.y0= beam_radius #initial ray height (mm) at the ENT surface
        self.u0= arctan((beam_radius-object_height)/Obj_2_Ent_tickness) #initial ray angle (rad) at the ENT surface
        self.y0_copy= self.y0
        self.u0_copy= self.u0

        self.Obj2ENT=Obj_2_Ent_tickness
        self.Z_center=[]
        z0=0
        for i, surf in enumerate(self.Surfaces):
            if i==0 or surf=='ims': pass
            else:
                z0+= self.Surfaces[surf].thickness
                self.Z_center.append(z0)


        self.u_marg= None
        self.y_marg= None
        self.active_stop= None
        self.Surfaces_copy= None
        # self.y_chief_marg= None
        self.AST_surface=None
        self.FOV_deg= None
        self.h_obj_chief=None
        self.y_chief=None


    def _default_solver_systm(self, n_rays:int=3, intersection_method='distance', tol=1e-10, Surfaces= None, u_rad=None, y_mm= None, h_obj=None, FOV_deg=None):
        if array([Surfaces==None]).all():
            Surfaces = self.Surfaces.copy(deep=True)
        
        surfaces_keys = Surfaces.keys().to_list()

        if h_obj==None and FOV_deg==None:
            h_obj= self.h_obj
        elif h_obj!=None and FOV_deg==None:
            h_obj = h_obj
        elif h_obj ==None and FOV_deg!=None:
            h_obj= self.obj_height_from_FOV(FOV_deg)
        else: 
            warn(f"Both h_obj and FOV_dg are not None, you allowed to pass either of them!", UserWarning)
            h_obj= self.h_obj


        if u_rad==None and y_mm== None:
            y0_new= self.y0
        else:
            if u_rad!=None:
                y0_new= h_obj+ (tan(u_rad)*self.Obj2ENT)
            else:
                y0_new= y_mm
    
        rays = -linspace(-1,1, n_rays, endpoint=True)
        Rays_Result=[]
        for ray in rays:

            y_init = y0_new*ray
            u_init = arctan((y_init-h_obj)/self.Obj2ENT)

            self.Y_inter= [y_init]
            self.U= [u_init]
            self.ERR= [0]
            self.Z_int= [0]
            self.Z= [0]
            self.dZ= [0]
            self.Z_act= [0]

            dz=0
            for i, surf in enumerate(surfaces_keys):
                if i==0 or surf==surfaces_keys[-1]: continue 

                surf_crnt= Surfaces[surf]
                surf_nxt= Surfaces[surfaces_keys[i+1]]
                y0= self.Y_inter[i-1]
                u0= self.U[i-1]
                z= surf_crnt.thickness
                z_act= z-dz
                y1,u1, z_int,err= self._solver(y0,u0,surf_crnt,surf_nxt,z_act, intersection_method=intersection_method, tol=tol)
                dz= z_int-z_act
                self.Y_inter.append(y1)
                self.U.append(u1)
                self.ERR.append(err)
                self.Z_int.append(z_int)
                self.Z.append(z)
                self.Z_act.append(z_act)
                self.dZ.append(dz)

            Results= DataFrame({'Z_mm':self.Z, 'Z_int_mm':self.Z_int,'Z_act_mm':self.Z_act,'dZ_mm':self.dZ,'Y_mm':self.Y_inter, 'U_rad':self.U, 'ERR':self.ERR}, index=Surfaces.keys().values[1:])
            # Results.index=[s.upper() for s in surfaces_keys[1:]]
            Rays_Result.append(Results)
        return Rays_Result



    def Solve_System(self, n_rays:int=3, solver:str='default',intersection_method='distance',ST_Active:bool=False,
                      tol=1e-10, marginal_ang_tol=1e-6, u_rad=None, y_mm=None,h_obj=None, FOV_deg=None, chief_guess=0):
        """
        General System Solver
        n_rays: the number of rays to pass for the fan field ray tracing
        solver: the type of tracing as following:
            - 'default'  : Fan field ray tracing (depends on the userdefined ray height at the entrance surface [ent] and number of rays [n_ray])
            - 'chief'    : Chief ray tracing calcultes automaticlly the angle of the ray that is needed to have the ray height= 0 at the AST surface
            - 'marginal' : Marginal ray tracing calcultes automaticlly the angle of the ray that is needed to have the ray height= AST Size (edge of the AST) at the AST surface
            - 'arb_angle':
        """
        if solver.lower()=='default':
            Rays_Result= self._default_solver_systm(n_rays=n_rays, intersection_method=intersection_method, tol=tol, u_rad=u_rad, y_mm=y_mm,h_obj=h_obj, FOV_deg=FOV_deg)
            return Rays_Result
        
        elif solver.lower()=='chief':
            # if FOV_deg==None:
            #     h_obj= self.h_obj
            # y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active)

            self._chief_solved_check(marginal_ang_tol=marginal_ang_tol, tol=tol, FOV_deg=FOV_deg, ST_Active=ST_Active, chief_guess=chief_guess)
            Rays_Result = self._default_solver_systm( n_rays=1,intersection_method=intersection_method,tol=tol, y_mm=self.y_chief,h_obj=self.h_obj_chief)
            return Rays_Result
            

        elif solver.lower()=='marginal':
            self._ast_solved_check(marginal_ang_tol=marginal_ang_tol, ST_Active=ST_Active)
            u= self.u_marg
            y= self.y_marg       

            nrys=n_rays
            if n_rays>2:
                nrys=2
            if self.Obj2ENT<1e4:
                Rays_Result= self._default_solver_systm(n_rays=nrys, intersection_method=intersection_method, tol=tol, u_rad=u, y_mm=y_mm,h_obj=0, FOV_deg=FOV_deg)
                y0= tan(u)*self.Obj2ENT
                self.y_marg = y0
            else:
                Rays_Result= self._default_solver_systm(n_rays=nrys, intersection_method=intersection_method, tol=tol, u_rad=u_rad, y_mm=y,h_obj=0, FOV_deg=FOV_deg)
                
                self.y_marg = y
            return Rays_Result
        

        elif solver.lower()=='arbitrary':
            if u_rad==None and y_mm==None:
                raise ValueError("No angle was passed u_rad is None and No height was passed y_mm is None")
            if u_rad!=None and y_mm!=None:
                raise ValueError(f"both arbitrary angle and arbitrary height were passed, pass only one of them! u_rad: {u_rad}, y_mm: {y_mm}")
            Rays_Result= self._default_solver_systm(n_rays, intersection_method, tol, u_rad=u_rad, y_mm=y_mm)
            if u_rad!=None:
                self.y_arb = self.h_obj+ (tan(u_rad)*self.Obj2ENT)
            else:
                self.y_arb = y_mm
        
            return Rays_Result

    def _ast_solved_check(self,marginal_ang_tol=1e-6, ST_Active:bool=False):
            # print(1)
            if all([self.u_marg==None , self.y_marg==None]) or ST_Active!=self.active_stop:
                # print(2)
                if self.Obj2ENT<1e4:
                    # print(3)
                    self.AST_surface,u= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                    self.u_marg=u
                else:
                    # print(4)
                    self.AST_surface,y= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                    self.y_marg=y
                # print(5)
                self.Surfaces_copy=self.Surfaces.copy(deep=True)
                self.active_stop= ST_Active
                return True
            
            else:
                # print(6)
                if ST_Active:
                    # print(7)
                    if not  self.Surfaces.iloc[:10,:].equals(self.Surfaces_copy.iloc[:10,:]):
                        # print(8)
                        if self.Obj2ENT<1e4:
                            # print(9)
                            self.AST_surface,u= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.u_marg=u
                        else:
                            # print(10)
                            self.AST_surface,y= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.y_marg=y
                        # print(11)
                        self.Surfaces_copy=self.Surfaces.copy(deep=True)
                        self.active_stop= ST_Active
                
                        return True


                else:
                    # print(12)

                    if not self.Surfaces.iloc[:8,:].equals(self.Surfaces_copy.iloc[:8,:]):
                        # print(13)
                        if self.Obj2ENT<1e4:
                            self.AST_surface,u= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.u_marg=u
                            # print(14)
                        else:
                            # print(15)
                            self.AST_surface,y= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.y_marg=y
                        # print(16)
                        self.Surfaces_copy=self.Surfaces.copy(deep=True)
                        self.active_stop= ST_Active
                        return True
            # print(17)
            return False


    def _chief_solved_check(self,marginal_ang_tol=1e-6, tol=1e-6, FOV_deg=None, ST_Active:bool=False,chief_guess=0):
        if self._ast_solved_check(ST_Active=ST_Active):
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0

        elif all([self.FOV_deg== None, self.h_obj_chief==None]) or self.y_chief==None:
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0
        
        elif self.FOV_deg!=FOV_deg:
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0
        
        elif self.FOV_deg==None and self.h_obj_chief!= self.h_obj:
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0

        


    def _plot_system(self,fig=None, ax=None, n_rays:int=3,ST_Active:bool=True,show_obj:bool=False, show_img:bool=True, 
                     lens_layout:bool=True, linestyle:str='-', ray_color:str='r',obj_col:str='orange', figsize=(10,4), 
                     Results=None, xscale:str='symlog',yscale:str='symlog',solver:str='default', intersection_method:str='distance', 
                     tol=1e-10,marginal_ang_tol=1e-6 , u_rad=None, y_mm=None, h_obj=None, FOV_deg=None, chief_guess=0):
        
        if lens_layout:
            fig,ax= self.Lens_Layout(fig, ax,show_obj, show_img,ST_Active,obj_col, figsize, xscale=xscale, yscale=yscale)
        elif fig==None or ax==None:
            fig,ax= subplots(figsize=figsize, tight_layout=True)

        Ray_Result=Results
        # print('Results', Results)
        if Results is None:
            Ray_Result= self.Solve_System(n_rays=n_rays,solver=solver, intersection_method=intersection_method,ST_Active=ST_Active,
                                          tol=tol, marginal_ang_tol=marginal_ang_tol, u_rad=u_rad, y_mm=y_mm, h_obj=h_obj, FOV_deg=FOV_deg,chief_guess=chief_guess)
        print(Ray_Result[0].Y_mm.values)
        dz= self.Obj2ENT
        if not show_obj:
            dz=0
        for result in Ray_Result:
            z_int_mm= cumsum(result.Z_int_mm.values)+dz
            y_mm=result.Y_mm.values
            if show_obj:
                h_obj= self.h_obj
                y_0= self.y0
                if solver.lower()== 'marginal':
                    h_obj=0
                    n_rays=2
                    y_0= self.y_marg
                    # print(h_obj, y_init)
                    
                if solver.lower() in ['chief']:
                    n_rays=1
                    h_obj= self.h_obj_chief
                    y_0= self.y_chief

                rays= -linspace(-1,1, n_rays, endpoint=True)
                for ray in rays:

                    y_init= y_0*ray

                    if solver.lower() in ['arbitrary']:
                        y_init= self.y_arb*ray

                    # print([0, z_int_mm[0]], [h_obj, y_init])
                    ax.plot([0, z_int_mm[0]], [h_obj, y_init], linestyle=linestyle,color=ray_color)
                
###############################################
            if ST_Active and solver.lower() not in ['marginal', 'm', 'marg']:
                z_int_mm, y_mm,_= self._check_stop(result, self.Surfaces.copy(deep=True), dz)

            print(y_mm)
            ax.plot(z_int_mm, y_mm, linestyle=linestyle,color=ray_color)
        
        return fig, ax, Ray_Result
    
    def Plot_System(self,fig=None, ax=None, n_rays:int=3, show_reference:bool=True,ST_Active:bool=True,show_obj:bool=False, show_img:bool=True, 
                    lens_layout:bool=True, linestyle:str='-', ray_color='r', ref_linestyle:str='--', ref_color:str='b',obj_col:str='orange', ref_obj_col:str='cyan', 
                    figsize=(10,4), Results=None, xscale:str='symlog',yscale:str='symlog',solver:str='default', intersection_method:str='distance', 
                    tol=1e-10, marginal_ang_tol=1e-6 , u_rad=None, y_mm=None, h_obj=None, FOV_deg=None,chief_guess=0):
        
        fig, ax, Ray_Result= self._plot_system(fig, ax, n_rays,ST_Active,show_obj, show_img, lens_layout, linestyle, ray_color, 
                                               obj_col,figsize, Results, xscale,yscale,solver, intersection_method, 
                                               tol, marginal_ang_tol=marginal_ang_tol, u_rad=u_rad, y_mm=y_mm, h_obj=h_obj, FOV_deg=FOV_deg, chief_guess=chief_guess)
        if self.h_obj>0 and show_reference:
            self.h_obj_copy= self.h_obj
            self.h_obj=0
            fig, ax, Ray_Result= self._plot_system(fig, ax, n_rays,ST_Active,show_obj, show_img, lens_layout, ref_linestyle, 
                                                   ref_color, ref_obj_col,figsize, Results, xscale,yscale,solver, 
                                                   intersection_method, tol=tol, marginal_ang_tol=marginal_ang_tol, u_rad=u_rad, 
                                                   y_mm=y_mm, h_obj=h_obj, FOV_deg=FOV_deg, chief_guess=chief_guess)
            self.h_obj=self.h_obj_copy
        return fig, ax, Ray_Result

    def _check_stop(self, result, Surfaces, dz=0):
        surface_res= Surfaces.keys().values
        z_int_mm= cumsum(result.Z_int_mm.values)+dz
        y_mm=result.Y_mm.values
        func= interp1d(z_int_mm, y_mm)

        z_mm= cumsum(result.Z_mm.tolist())+dz

        STs=Surfaces.loc['stop',:][1:].tolist()
        ST_sizes=Surfaces.loc['stop_size',:][1:].tolist()

        surf_keys=(Surfaces.keys().tolist())[1:]
        Z_ap= []
        Ss= []
        for j,(st,zap) in enumerate(zip(STs,z_mm)):
            if st:
                Z_ap.append(zap) 
                Ss.append(Surfaces[surf_keys[j]])

        for (S,z_ap) in zip(Ss, Z_ap):
            R= S.radius
            surface_name=S.surface_name
            t= S.thickness
            ap= S.aperture
            mat= S.material
            typ= S.type
            k= S.conic
            As= S.A_coefficient
            ast= S.ast
            stop= S.stop
            stop_size= S.stop_size
            
            yyy= abs(R)-1e-10
            if ap<abs(R):
                yyy= ap
            if not any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                if typ=='asph':
                    z_ap= self.sag_aspherical(yyy, R, k, As, z_ap)

                else:
                    z_ap= self.sag_spherical(yyy, R, z_ap)

            y_ap= func(z_ap)

            if abs(y_ap)> abs(stop_size):
                z_int_mm= [i for i in z_int_mm if i <=z_ap]
                surface_res= [surf_keys[w] for w,i in enumerate(z_int_mm) if i <=z_ap]
                len_diff= len(surf_keys)- len(z_int_mm)
                if z_ap not in z_int_mm:
                    z_int_mm+=[z_ap]
                    surface_res+= [S.name]
                if len_diff>0:
                    for i in range(len_diff):
                        z_int_mm+=[nan]

                y_mm= func(z_int_mm)
                break 
        # print(z_int_mm, y_mm, surface_res)
        return z_int_mm, y_mm, surface_res

    def find_max_angle(self, tol=1e-6, ST_Active:bool=False):
        """
        Find the maximum input ray angle (u_rad) that still reaches the ims surface.
        Uses _default_solver_systm to check if the ray reaches ims (Y_mm is not nan).
        """
        Surfaces= self.Surfaces.copy(deep=True)
        y_ap=1e20

        if tol<1e-15:
            tol=1e-15

        tol_copy=tol
        if ST_Active:
            tol=1e-3

        while True:

            low = 0.0
            high = arctan(Surfaces.surf2.aperture/self.Obj2ENT)+.2
            max_angle = 0.0
            err=1e20
            while err >tol :  # desired precision
                mid = (low + high) / 2
                try:
                    Result = self._default_solver_systm(n_rays=1, Surfaces=Surfaces,u_rad=mid, h_obj=0)
                    
                    # Check if ray reached ims: Y_mm is a valid number
                    reached_ims = not isnan(Result[0].Y_mm.values[-1])
                except:
                    reached_ims = False

                if reached_ims:
                    max_angle = mid  # ray reached ims, try bigger angle
                    low = mid
                else:

                    high = mid  # ray blocked, try smaller angle

                err= rad2deg(high) - rad2deg(low)
            if isnan(Result[0].Y_mm.values[-1]):
                tol*=10

            else:
                break


        Y_ap=[]
        S_ap=[]
        ERR_ap=[]
        
        if ST_Active:
            max_angle_res= max_angle+.1
            tol= tol_copy
            
            while True:

                low = 0.0
                high = max_angle_res

                max_angle = 0.0
                err=1e20
                while err >tol :  # desired precision
                    mid = (low + high) / 2
                    try:
                        Result = self._default_solver_systm(n_rays=1, Surfaces=Surfaces,u_rad=mid, h_obj=0)
                        
                        # Check if ray reached ims: Y_mm is a valid number
                        reached_ims = not isnan(Result[0].Y_mm.values[-1])
                    except:
                        reached_ims = False

                    stat= True
                    result=Result[0]
                    z_int_mm= cumsum(result.Z_int_mm.values)
                    y_mm=result.Y_mm.values
                    func= interp1d(z_int_mm, y_mm)

                    z_mm= cumsum(result.Z_mm.tolist())
                    # z_mm= cumsum(RT.Surfaces.loc['thickness', :].values)[:-1].tolist()
                    STs=Surfaces.loc['stop',:][1:].tolist()
                    ST_sizes=Surfaces.loc['stop_size',:][1:].tolist()
                    # print(ASTs)
                    surf_keys=(Surfaces.keys().tolist())[1:]
                    Z_ap= []
                    Ss= []
                    Ss_keys= []
                    for j,(st,zap) in enumerate(zip(STs,z_mm)):
                        if st:
                            Z_ap.append(zap) 
                            Ss.append(Surfaces[surf_keys[j]])
                            Ss_keys.append(surf_keys[j])
                    
                    Y_ap=[]
                    S_ap=[]
                    ERR_ap=[]


                    for (S,z_ap, s_key) in zip(Ss, Z_ap,Ss_keys):

                        R= S.radius
                        t= S.thickness
                        ap= S.aperture
                        mat= S.material
                        typ= S.type
                        k= S.conic
                        As= S.A_coefficient
                        ast= S.ast
                        stop= S.stop
                        stop_size= S.stop_size
                        
                        
                        # print(z_ap)
                        yyy= abs(R)-1e-10
                        if ap<abs(R):
                            yyy= ap
                        if not any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                            if typ=='asph':
                                z_ap= self.sag_aspherical(yyy, R, k, As, z_ap)
                                # print(z_ap)

                            else:
                                z_ap= self.sag_spherical(yyy, R, z_ap)
                                # print(z_ap)

                    
                        y_ap= func(z_ap)

                        Y_ap.append(y_ap)
                        S_ap.append(s_key)
                        ERR_ap.append(abs(abs(y_ap)-abs(stop_size)))

                        if abs(y_ap)> abs(stop_size):
                            stat= False
                            break   



                    if reached_ims and stat:
                        max_angle = mid  # ray reached ims, try bigger angle
                        low = mid
                    else:

                        high = mid  # ray blocked, try smaller angle

                    err= rad2deg(high) - rad2deg(low)

                if isnan(Result[0].Y_mm.values[-1]):
                    tol*=10

                else:
                    break

        return max_angle, Result, Y_ap, S_ap, ERR_ap


    def find_max_height_infinity(self, tol=1e-6, ST_Active:bool=False):
        """
        Find the maximum input ray height and angle (u_rad) for ray coming from infinity that still reaches the ims surface.
        Uses _default_solver_systm to check if the ray reaches ims (Y_mm is not nan).
        """

        Surfaces= self.Surfaces.copy(deep=True)
        y_ap=1e20

        if tol<1e-15:
            tol=1e-15

        tol_copy=tol
        if ST_Active:
            tol=1e-3

        while True:

            low = 0.0
            high = Surfaces.surf2.aperture
            max_height = 0.0
            err=1e20
            while err >tol :  # desired precision

                mid = (low + high) / 2
                try:
                    Result = self._default_solver_systm(n_rays=1, Surfaces=Surfaces,y_mm=mid, h_obj=0)
                    
                    # Check if ray reached ims: Y_mm is a valid number
                    reached_ims = not isnan(Result[0].Y_mm.values[-1])
                except:
                    reached_ims = False

                if reached_ims:
                    max_height = mid  # ray reached ims, try bigger angle
                    low = mid
                else:

                    high = mid  # ray blocked, try smaller angle

                err= high - low
            if isnan(Result[0].Y_mm.values[-1]):
                tol*=10

            else:
                break


        Y_ap=[]
        S_ap=[]
        ERR_ap=[]
        
        if ST_Active:
            max_height_res= max_height+.5
            tol= tol_copy
            
            while True:

                low = 0.0
                high = max_height_res

                max_height = 0.0
                err=1e20
                while err >tol :  # desired precision
                    mid = (low + high) / 2
                    try:
                        Result = self._default_solver_systm(n_rays=1, Surfaces=Surfaces,y_mm=mid, h_obj=0)
                        
                        # Check if ray reached ims: Y_mm is a valid number
                        reached_ims = not isnan(Result[0].Y_mm.values[-1])
                    except:
                        reached_ims = False

                    stat= True
                    result=Result[0]
                    z_int_mm= cumsum(result.Z_int_mm.values)
                    y_mm=result.Y_mm.values
                    func= interp1d(z_int_mm, y_mm)

                    z_mm= cumsum(result.Z_mm.tolist())
                    # z_mm= cumsum(RT.Surfaces.loc['thickness', :].values)[:-1].tolist()
                    STs=Surfaces.loc['stop',:][1:].tolist()
                    ST_sizes=Surfaces.loc['stop_size',:][1:].tolist()
                    # print(ASTs)
                    surf_keys=(Surfaces.keys().tolist())[1:]
                    Z_ap= []
                    Ss= []
                    Ss_keys= []
                    for j,(st,zap) in enumerate(zip(STs,z_mm)):
                        if st:
                            Z_ap.append(zap) 
                            Ss.append(Surfaces[surf_keys[j]])
                            Ss_keys.append(surf_keys[j])
                    
                    Y_ap=[]
                    S_ap=[]
                    ERR_ap=[]


                    for (S,z_ap, s_key) in zip(Ss, Z_ap,Ss_keys):

                        R= S.radius
                        t= S.thickness
                        ap= S.aperture
                        mat= S.material
                        typ= S.type
                        k= S.conic
                        As= S.A_coefficient
                        ast= S.ast
                        stop= S.stop
                        stop_size= S.stop_size
                        
                        
                        # print(z_ap)
                        yyy= abs(R)-1e-10
                        if ap<abs(R):
                            yyy= ap
                        if not any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                            if typ=='asph':
                                z_ap= self.sag_aspherical(yyy, R, k, As, z_ap)
                                # print(z_ap)

                            else:
                                z_ap= self.sag_spherical(yyy, R, z_ap)
                                # print(z_ap)

                    
                        y_ap= func(z_ap)

                        Y_ap.append(y_ap)
                        S_ap.append(s_key)
                        ERR_ap.append(abs(abs(y_ap)-abs(stop_size)))

                        if abs(y_ap)> abs(stop_size):
                            stat= False
                            break   



                    if reached_ims and stat:
                        max_height = mid  # ray reached ims, try bigger angle
                        low = mid
                    else:

                        high = mid  # ray blocked, try smaller angle

                    err= high - low

                if isnan(Result[0].Y_mm.values[-1]):
                    tol*=10

                else:
                    break
            
        return max_height, Result, Y_ap, S_ap, ERR_ap
      
    
    def find_AST(self, tol=1e-6, ST_Active:bool=False):
        Surfaces= self.Surfaces.copy(deep=True)
        if self.Obj2ENT<1e4:
            
            max_angle, Result, Y_ap, S_ap,ERR_ap = self.find_max_angle(tol=tol,ST_Active=ST_Active)

            err= min(abs(abs(Surfaces.iloc[3,:].values[1:])-abs(Result[0].Y_mm.values)))
            # idx_ast= argmin(abs(abs(Surfaces.iloc[3,:].values[1:])-abs(Result[0].Y_mm.values)))
            # self.AST_surface= Surfaces.keys()[int(idx_ast)+1]
            # print(max_angle+1e-3)
            Rays_Result= self._default_solver_systm(n_rays=1, u_rad=max_angle+1e-3,h_obj=0)
            if not ST_Active:
                # print(1, Rays_Result[0].Y_mm.values)
                idx_ast= where(isnan(Rays_Result[0].Y_mm.values))[0][0]
                self.AST_surface= Rays_Result[0].index.values[int(idx_ast-1)]

            else:
                # print(2, Rays_Result[0].Y_mm.values)
                # print(Rays_Result[0])
                # print(Rays_Result[0])
                _, y_mm, surfaces= self._check_stop( Rays_Result[0], Surfaces)
                # print(y_mm)
                idx_ast= where(isnan(y_mm))[0][0]
                self.AST_surface= surfaces[int(idx_ast-1)]
            
            if self.AST_surface=='ent':
                if not Surfaces.ent.stop:
                    self.AST_surface='surf2'
                else:
                    Rays_Result= self._default_solver_systm(n_rays=1, u_rad=max_angle,h_obj=0)
                    y_ent= Rays_Result[0].Y_mm.values[0]
                    y_surf2= Rays_Result[0].Y_mm.values[1]
                    err_ent=abs(abs(y_ent)-abs(Surfaces.ent.stop_size))
                    
                    if Surfaces.surf2.stop:
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.stop_size))
                    else:
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.aperture))
                    
                    if err_ent>err_surf2:
                        self.AST_surface='surf2'




            #     Rays_Result= self._default_solver_systm(n_rays=1, u_rad=max_angle,h_obj=0)
            #     if Surfaces.ent.stop:

            return self.AST_surface, max_angle
        
        else:
            max_height, Result, Y_ap, S_ap, ERR_ap   = self.find_max_height_infinity(tol=tol,ST_Active=ST_Active)
            
            err= min(abs(abs(Surfaces.iloc[3,:].values[1:])-abs(Result[0].Y_mm.values)))
            # idx_ast= argmin(abs(abs(Surfaces.iloc[3,:].values[1:])-abs(Result[0].Y_mm.values)))
            # self.AST_surface= Surfaces.keys()[int(idx_ast)+1]

            Rays_Result= self._default_solver_systm(n_rays=1, y_mm=max_height+1e-3,h_obj=0)
            if not ST_Active:
                idx_ast= where(isnan(Rays_Result[0].Y_mm))[0][0]
                self.AST_surface= Rays_Result[0].index.values[int(idx_ast-1)]

            else:
                _, y_mm, surfaces= self._check_stop( Rays_Result[0], Surfaces)
                self.AST_surface= surfaces[-1]


            if self.AST_surface=='ent':
                if not Surfaces.ent.stop:
                    self.AST_surface='surf2'
                else:
                    Rays_Result= self._default_solver_systm(n_rays=1, y_mm=max_height,h_obj=0)
                    y_ent= Rays_Result[0].Y_mm.values[0]
                    y_surf2= Rays_Result[0].Y_mm.values[1]
                    err_ent=abs(abs(y_ent)-abs(Surfaces.ent.stop_size))
                    
                    if Surfaces.surf2.stop:
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.stop_size))
                    else:
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.aperture))
                    
                    if err_ent>err_surf2:
                        self.AST_surface='surf2'

        

            return self.AST_surface, max_height


    def FOV_from_obj_height(self, h_obj):
        """
        Compute the FOV (field of view half-angle, in degrees) 
        from a given object height.

        Parameters
        ----------
        h_obj : float
            Object height at the obj surface.

        Returns
        -------
        FOV_deg : float
            Half-field angle in degrees.
        """
        if self.Obj2ENT == 0:
            raise ValueError("Obj2ENT (object to entrance pupil distance) is zero → cannot compute FOV.")

        FOV_rad = arctan(h_obj / self.Obj2ENT)
        FOV_deg = rad2deg(FOV_rad)
        return FOV_deg

    def max_fov_deg(self):
        """
        Compute the maximum possible FOV (degrees) based on entrance pupil aperture.

        Returns
        -------
        max_fov_deg : float
            Maximum half-field angle in degrees
        """
        # Entrance pupil radius
        y_max = self.Surfaces.surf2.aperture  

        # Distance from object to entrance pupil
        L_obj = self.Obj2ENT  

        if L_obj <= 0:
            raise ValueError("Object distance (OBJ2Ent) must be positive.")

        # Half field angle (one side)
        theta_max = arctan(y_max / L_obj)  # radians
        return degrees(theta_max)


    def chief_ray_guess(self, fov_deg):
        """
        Provide an initial guess for chief ray height at entrance pupil,
        based on requested FOV.

        Parameters
        ----------
        fov_deg : float
            Desired half-field angle (deg).

        Returns
        -------
        y_guess : float
            Initial guess for chief ray height at entrance pupil
        """
        max_fov = self.max_fov_deg()
        y_max = self.Surfaces.surf2.aperture

        if abs(fov_deg) > max_fov:
            print(f"⚠️ Warning: Requested FOV={fov_deg:.2f}° exceeds system limit {max_fov:.2f}°")

        # Scale guess linearly with ratio of requested FOV to max FOV
        y_guess = (fov_deg / max_fov) * y_max
        return y_guess


    def obj_height_from_FOV(self, FOV_deg):
        """
        Compute object height y_obj from the desired FOV (field of view).
        
        Parameters
        ----------
        FOV_deg : float
            Desired field of view in degrees (half-angle).

            System Type                        Typical FOV (full angle)
        Microscope Objective                         0.5° - 10°
        Camera Lens (wide-angle)                     60° - 120°
        Standard Camera Lens (35-50 mm)              40° - 60°
        Telephoto Lens                               10° - 30°
        Telescope / Collimated System                < 1°
        Endoscope / Small Optics                     20° - 90°

       
        
        object distance, in mm:
        infinity):        ~ 0.1° - 10°
        10,000 ≤ L < 1e6 (very far):            ~ 0.5° - 20°
        2,000 ≤ L < 10,000 (far):               ~ 1° - 30°
        300 ≤ L < 2,000 (mid):                  ~ 2.5° - 45°
        50 ≤ L < 300 (near):                    ~ 5° - 60°
        L < 50 (macro/very near):               ~ 10° - 70°


        Returns
        -------
        y_obj : float
            Object height at the obj surface corresponding to the given FOV.
        """
        FOV_rad = deg2rad(FOV_deg)  # convert degrees to radians
        y_obj = self.Obj2ENT* tan(FOV_rad)
        return y_obj

    def find_max_height_chief(self, fov_deg=10, marginal_ang_tol=1e-6, tol=1e-3, ST_Active: bool=False, chief_guess=0):
        """
        Find the entrance pupil ray height that produces the chief ray
        (crossing the aperture stop at y=0).
        """

        # Ensure AST position solved
        self._ast_solved_check(marginal_ang_tol=marginal_ang_tol, ST_Active=ST_Active)

        y_max = self.Surfaces.surf2.aperture
        h_obj= self.h_obj
        if fov_deg!=None:
            h_obj = self.obj_height_from_FOV(FOV_deg=fov_deg)
        

        Surfaces = self.Surfaces.copy(deep=True)
        surf_keys = Surfaces.keys().values[1:]  # skip obj
        ast_idx = where(surf_keys == self.AST_surface)[0][0]

        # print(ast_idx)
        if ast_idx ==0:
            return 0, h_obj, 0

        if ast_idx ==1:
            u= arctan((0-h_obj)/(self.Obj2ENT+Surfaces.ent.thickness))
            # print(Surfaces.ent.thickness*tan(abs(u)))
            return Surfaces.ent.thickness*tan(abs(u)), h_obj, 0

        def ERR(y0, h_obj, n_rays=1, intersection_method='distance', tol=1e-10):
            try:
                Result = self._default_solver_systm(
                    n_rays=n_rays,
                    intersection_method=intersection_method,
                    tol=tol,
                    Surfaces=Surfaces,
                    y_mm=float(y0),
                    h_obj=h_obj
                )
                # ray blocked? last Y_mm is nan
                if isnan(Result[0].Y_mm.values[-1]):
                    return 1e6  # heavy penalty
                y_ast = Result[0].Y_mm.values[ast_idx]
                # print(y0, y_ast, h_obj)
                return y_ast**2  # squared error → smooth near 0
            except Exception:
                return 1e6  # penalty on failure

        # Wrap for scipy
        func = lambda y: ERR(y[0], h_obj=h_obj)

        bnds = [(-y_max, y_max)]
        # X0= [0,1,3,6, 9, 10, 15]
        # if y0_guess not in X0:
        #     X0= [y0_guess]+X0
        # Y_res=[]
        # err_res=[]
        # for x0 in X0:
        #     res = minimize(func, x0=x0, tol=tol, method='SLSQP', bounds=bnds)
        #     Y_res.append(res.x[0])
        #     err_res.append(res.fun)

        # return Y_res[argmin(err_res)], h_obj, err_res[argmin(err_res)]

        # x0=chief_guess
        # err_res=1e20
        # while err_res>tol:
        #     res = minimize(func, x0=x0, tol=tol, method='SLSQP', bounds=bnds)
        #     y_res= (res.x[0])
        #     err_res= (res.fun)
        #     x0+=2
        # return y_res, h_obj, err_res

        X0= [chief_guess,self.chief_ray_guess(self.FOV_from_obj_height(h_obj))]

        Y_res=[]
        err_res=[]
        for x0 in X0:
            res = minimize(func, x0=x0, tol=tol, method='SLSQP', bounds=bnds)
            Y_res.append(res.x[0])
            err_res.append(res.fun)
            if res.fun<tol:
                return res.x[0], h_obj, res.fun
                
        

        return Y_res[argmin(err_res)], h_obj, err_res[argmin(err_res)]



    # def find_max_height_chief(self, fov_deg=10, marginal_ang_tol=1e-6, tol=1e-3, ST_Active: bool=False):
    #     """
    #     Find the maximum input ray height (at entrance pupil) that, for a given object height,
    #     produces a chief ray passing through the AST surface at y=0.

    #     Parameters
    #     ----------
    #     fov_deg : float
    #         Field of view angle in degrees.
    #     marginal_ang_tol : float
    #         Tolerance for marginal ray solver.
    #     tol : float
    #         Tolerance for chief ray intersection with AST (in mm).
    #     ST_Active : bool
    #         Whether to enforce stop surfaces.

    #     Returns
    #     -------
    #     max_height : float
    #         Ray entrance height at 'ent' surface that makes chief ray pass AST at y=0
    #     h_obj : float
    #         Object height corresponding to FOV
    #     Result : DataFrame
    #         Raytrace of the chief ray (or best approximation if no exact hit)
    #     """

   
    #     self._ast_solved_check(marginal_ang_tol=marginal_ang_tol, ST_Active=ST_Active)
    #     y_max= self.Surfaces.surf2.aperture


    #     h_obj = self.obj_height_from_FOV(FOV_deg=fov_deg)


    #     Surfaces = self.Surfaces.copy(deep=True)
    #     surf_keys = Surfaces.keys().values[1:]  # skip 'obj'
    #     ast_idx = where(surf_keys == self.AST_surface)[0][0]  # AST index in dataframe

    #     def ERR(y0, h_obj,  n_rays: int = 1, intersection_method: str = 'distance', tol: float = 1e-10):
    #         Result = self._default_solver_systm(n_rays=n_rays, intersection_method= intersection_method, tol=tol, y_mm=y0, h_obj=h_obj)
    #         return abs(Result[0].Y_mm.values[ast_idx])
        
    #     func = lambda y: ERR(y[0], h_obj=h_obj, n_rays=1, intersection_method = 'distance', tol = 1e-10)

    #     bnds = [(-y_max,y_max)]
    #     ['Nelder-Mead','L-BFGS-B','SLSQP']
    #     res = minimize(func, x0=[-y_max/4], tol=tol, method='SLSQP', bounds=bnds)
        
    #     # Return best approximation if exact solution not reached
    #     return res.x, h_obj, res.fun



        # def _chief_syst(self):

        #     Surfaces= self.Surfaces.copy(deep= True)
        #     Surfaces2= self.Surfaces.copy(deep= True)
        #     for l,s in enumerate(Surfaces):
        #         if Surfaces[s].stop:
        #             surf= Surfaces2[s]
        #             R= surf.radius
        #             mat= surf.material
        #             t= surf.thickness
        #             surf.stop=False
        #             stop_size=surf.stop_size
        #             typ= surf.type
        #             k= surf.conic
        #             As= surf.A_coefficient

        #             # if any([(R<0 and mat.lower()!='air'),(R>=0 and mat.lower()!='air')]):
        #             #     t_surf= 0
        #             #     s=int(where(Surfaces2.keys().values==surf.surface_name)[0][0])+1
        #             #     if s!='ent':
        #             #         idx= int(where(Surfaces2.keys().values==surf.surface_name)[0][0])

        #             # else:
        #             #     t_surf=t
        #             #     surf.thickness=0
        #             #     idx=int(where(Surfaces2.keys().values==surf.surface_name)[0][0])+1
                        

        #             if any([(R<=0 and mat.lower()!='air'),(R>=0 and mat.lower()!='air')]):
        #                 t_surf=t
        #                 surf.thickness=0
        #                 idx=int(where(Surfaces2.keys().values==surf.surface_name)[0][0])+1

        #             else:
        #                 yyy= abs(R)-1e-10
        #                 if stop_size<abs(R):
        #                     yyy= stop_size
        #                 if typ=='asph':
        #                     t_surf= self.sag_aspherical(yyy, R, k, As, 0)
        #                 else:
        #                     t_surf= self.sag_spherical(yyy, R, 0)#
                        
        #                 idx=int(where(Surfaces2.keys().values==surf.surface_name)[0][0])+1
        #                 Surfaces2[Surfaces2.keys().values[idx-1]].thickness-=t_surf
                    
        #             # surf.thickness=0
        #             add_surface= {'surface_name':f'{surf.surface_name}_ast',
        #                         'radius': 0,
        #                         'thickness': t_surf,
        #                         'aperture': stop_size, 
        #                         'material': 'air', 
        #                         'type': 'sph',
        #                         'conic': 0,
        #                         'A_coefficient': [],
        #                         'stop':True,
        #                         'stop_size':stop_size,
        #                         'ast':False}

        #             Surfaces2.insert(idx, f'{surf.surface_name}_ast', add_surface)
                    
        #     return Surfaces2




        # def _chief_marginal_solver_systm(self,u , intersection_method='distance', tol=1e-10):
        #     self.y0 = tan(u)*self.Obj2ENT+self.h_obj
        #     self.u0=u
        #     Result= self._default_solver_systm(1, intersection_method, tol)
        #     self.y0= self.y0_copy
        #     self.u0= self.u0_copy

        #     return Result
            

        # def find_ray_to_ast(self, y_target, u0_guess=0.15,intersection_method='distance', tol=1e-10, maxiter=50):
        #     """
        #     Solve for entrance angle that makes ray hit y_target at the AST surface.
        #     Returns entrance height (y0 at ENT) and entrance angle u0 (rad).
        #     """
        #     Surfaces=self.Surfaces.copy(deep=True)
        #     ast_idx= int(argwhere(Surfaces.iloc[-1,:].values)[0][0])

        #     Surfaces_AST= Surfaces.iloc[:, :ast_idx].copy(deep=True)
        #     Surfaces_AST['AST']={'surface_name':'AST', 'radius':0, 'thickness':0, 'aperture':self.AST_size, 'material':'air', 'type':'sph', 'conic':0, 'A_coefficient':0, 'ast':False}
            

        #     def f(u0):

        #         Surfaces=Surfaces_AST.copy(deep=True)           
        #         surfaces_keys= Surfaces.keys().values
        #         Y_inter= [self.h_obj + self.Obj2ENT * tan(u0)]
        #         U= [u0]
        #         ERR= [0]
        #         Z_int= [0]
        #         Z= [0]
        #         dZ= [0]
        #         Z_act= [0]
        #         dz=0

        #         for i, surf in enumerate(surfaces_keys):
                    
        #             if surf=='obj' or surf=='AST': continue 
        #             surf_crnt = Surfaces[surf]
        #             surf_nxt  = Surfaces[surfaces_keys[i+1]]

        #             y0= Y_inter[i-1]
        #             u0= U[i-1]
        #             z= surf_crnt.thickness
        #             z_act= z-dz
        #             y1,u1, z_int,err= self._solver(y0,u0,surf_crnt,surf_nxt,z_act, intersection_method=intersection_method, tol=tol)
        #             dz= z_int-z_act
        #             Y_inter.append(y1)
        #             U.append(u1)
        #             ERR.append(err)
        #             Z_int.append(z_int)
        #             Z.append(z)
        #             Z_act.append(z_act)
        #             dZ.append(dz)

        #         y= Y_inter[-1]

        #         return y - y_target
            
        #     failure_reason=None
        #     def check_solution(u, method_name):
        #         err = abs(f(u))
        #         if err < tol:
        #             return u
        #         else:
        #             nonlocal failure_reason
        #             failure_reason = f"{method_name} found angle but error too large ({err:.3e})"
        #             return nan


        #     u_res=nan
        #     start_guesses = [u0_guess, 0.0, 0.05, -0.05, 0.1, -0.1,.2,-.2,1,-1,2,-2]
        #     for guess in start_guesses:
        #         try:
        #             u_sol = newton(f, guess, tol=tol, maxiter=maxiter)
        #             checked = check_solution(u_sol, f"Newton start={guess}")
        #             if not isnan(checked):
        #                 u_res= checked
        #                 break
        #         except RuntimeError:
        #             continue

        #     # recompute corresponding y0 at entrance
        #     y0 = self.h_obj + self.Obj2ENT * tan(u_res)
        #     return y0, u_res





        # def _optimizer(self, x0, surf2optimize, y_target, surf_target,parameter='thickness', intersection_method='distance', tol=1e-10,n_rays=21):
        #     """
        #     Objective function for optimization.
        #     Returns signed error (not RMS) between ray height at target and desired y_target.
        #     """

        #     Surfaces = self.Surfaces.copy(deep=True)
        #     surfaces_keys = list(Surfaces.keys())

        #     # --- Find indices ---
        #     try:
        #         i_opt = surfaces_keys.index(surf2optimize.lower())
        #     except ValueError:
        #         raise ValueError(f"{surf2optimize} surface was not found.")
        #     try:
        #         i_target = surfaces_keys.index(surf_target.lower())-1
        #     except ValueError:
        #         raise ValueError(f"{surf_target} surface was not found.")

        #     # --- Save original parameter ---
        #     surf_obj = Surfaces[surfaces_keys[i_opt]]
        #     original_val = getattr(surf_obj, parameter)

        #     # --- Apply trial value ---
        #     setattr(surf_obj, parameter, x0)

        #     # --- Trace rays ---
        #     Ray_Result = self._default_solver_systm(
        #         n_rays=n_rays,  # chief + marginal enough
        #         intersection_method=intersection_method,
        #         tol=tol,
        #         Surfaces=Surfaces
        #     )

        #     # pick the chief ray (center)
        #     mid = len(Ray_Result) // 2
        #     y_at_target = Ray_Result[mid]['Y_mm'].iloc[i_target]

        #     # --- Restore original value ---
        #     setattr(surf_obj, parameter, original_val)

        #     # --- Return signed error ---
        #     return y_at_target - y_target


        # def Optimizer(self, surf2optimize, y_target, surf_target,
        #             parameter='thickness', intersection_method='distance',
        #             tol=1e-10, x_guess=0.0, verbose=True, n_rays=21):
        #     """
        #     Robust surface optimizer: adjusts thickness/radius to achieve target ray height.
        #     """

        #     func = lambda x: self._optimizer(x, surf2optimize, y_target, surf_target,
        #                                     parameter, intersection_method, tol,n_rays=n_rays)

        #     failure_reason = None

        #     # ---- 1. Try Newton ----
        #     try:
        #         sol = newton(func, x0=x_guess, tol=1e-12, maxiter=50)
        #         if abs(func(sol)) < tol:
        #             return sol, func(sol) + y_target
        #     except Exception as e:
        #         failure_reason = f"Newton failed: {e}"

        #     # ---- 2. Try minimize_scalar ----
        #     try:
        #         res = minimize_scalar(lambda x: abs(func(x)),
        #                             bracket=(x_guess-5, x_guess+5))
        #         if res.success and abs(func(res.x)) < tol:
        #             return res.x, func(res.x) + y_target
        #     except Exception as e:
        #         failure_reason = f"minimize_scalar failed: {e}"

        #     # ---- 3. Try Brent root-finder ----
        #     try:
        #         sol = brentq(func, x_guess-5, x_guess+5, xtol=1e-12, maxiter=200)
        #         if abs(func(sol)) < tol:
        #             return sol, func(sol) + y_target
        #     except Exception as e:
        #         failure_reason = f"brentq failed: {e}"

        #     if verbose:
        #         print(f"⚠️ Optimizer failed. Last reason: {failure_reason}")

        #     return nan, nan


        # def Auto_Focus(self,n_rays=21, intersection_method='distance', tol=1e-10):
        #     Ray_Result = self._default_solver_systm(n_rays=n_rays, intersection_method=intersection_method, tol=tol)

        #     y_mm= [Ray_Result[i]['Y_mm'].tolist()[-1] for i in range(len(Ray_Result))]
        #     u_rad= [Ray_Result[i]['U_rad'].tolist()[-1] for i in range(len(Ray_Result))]

        #     z_mm= []
        #     for (y,u) in zip(y_mm, u_rad):
        #         if u!=0:
        #             z_res= -y/tan(u)
        #             if not isnan(z_res):
        #                 z_mm.append(z_res)




        #     Surfaces=self.Surfaces.copy(deep=True)
        #     Surfaces.ims.thickness= mean(z_mm)
        #     Ray_Result = self._default_solver_systm(n_rays=n_rays, intersection_method=intersection_method, tol=tol, Surfaces=Surfaces)
        #     y_mm= [Ray_Result[i]['Y_mm'].tolist()[-1] for i in range(len(Ray_Result))]
        #     return mean(z_mm),mean(y_mm)


        # def _ast_backward_trace(self,u0_guess,y_start,intersection_method='distance', tol=1e-10):
        #     Surfaces=self.Surfaces.copy(deep=True)
        #     ast_idx= int(argwhere(Surfaces.iloc[-1,:].values)[0][0])

        #     Surfaces_AST= Surfaces.iloc[:, :ast_idx+1].copy(deep=True) 
        #     thick= Surfaces_AST.iloc[2, :].values.tolist()
        #     thick.pop()
        #     thick.insert(0, 0)
        #     Surfaces_AST.iloc[2, :]= thick

        #     mat= Surfaces_AST.iloc[4, :].values.tolist()
        #     mat.pop()
        #     mat.insert(0, 'air')
        #     Surfaces_AST.iloc[4, :]= mat

        #     Rad= -Surfaces_AST.iloc[1, :].values
        #     Surfaces_AST.iloc[1, :]= Rad

        #     As= [(-array(a)).tolist() for a in Surfaces_AST.iloc[7, :].values]
        #     Surfaces_AST.iloc[7, :]= As
        #     Surfaces_AST['AST']={'surface_name':'AST', 'radius':0, 'thickness':0, 'aperture':self.AST_size, 'material':'air', 'type':'sph', 'conic':0, 'A_coefficient':0, 'ast':False}
        #     Surfaces_AST.obj.type='sph'
        #     Surfaces_AST= Surfaces_AST.iloc[:,::-1]
        #     # return Surfaces_AST
        #     def _trace( u0, y_start, intersection_method=intersection_method, tol=tol):
        #         """
        #         Trace a single ray from object to AST, return difference between ray height at AST and target.
        #         """
        #         Surfaces=Surfaces_AST.copy(deep=True)           
        #         surfaces_keys= Surfaces.keys().values
        #         Y_inter= [y_start]
        #         U= [u0]
        #         ERR= [0]
        #         Z_int= [0]
        #         Z= [0]
        #         dZ= [0]
        #         Z_act= [0]
        #         dz=0


        #         # Trace ray until AST surface
        #         # print(surfaces_keys)
        #         for i, surf in enumerate(surfaces_keys):
                    
        #             if surf=='obj': continue 
        #             surf_crnt = Surfaces[surf]
        #             surf_nxt  = Surfaces[surfaces_keys[i+1]]

        #             # print(i, surf_crnt, surf_nxt)

        #             y0= Y_inter[i-1]
        #             u0= U[i-1]
        #             z= surf_crnt.thickness
        #             z_act= z-dz
        #             y1,u1, z_int,err= self._solver(y0,u0,surf_crnt,surf_nxt,z_act, intersection_method=intersection_method, tol=tol)
        #             dz= z_int-z_act
        #             Y_inter.append(y1)
        #             U.append(u1)
        #             ERR.append(err)
        #             Z_int.append(z_int)
        #             Z.append(z)
        #             Z_act.append(z_act)
        #             dZ.append(dz)
        #             # print(i, surf)

        #         # print(U)
        #         res_index= Surfaces.keys().values
        #         result= DataFrame({'Z_mm':Z, 'Z_int_mm':Z_int,'Z_act_mm':Z_act,'dZ_mm':dZ,'Y_mm':Y_inter, 'U_rad':U, 'ERR':ERR}, index=res_index)
        #         return result


        #     def _trace_err(u0):
        #         result= _trace( u0=u0, y_start=y_start, intersection_method=intersection_method, tol=tol)
        #         y= result.Y_mm.values.tolist()[-1]
        #         return (y - self.h_obj)  
            

        #     func = lambda u: _trace_err( u)


        #     failure_reason = None

        #     def check_solution(u, method_name):
        #         err = abs(func(u))
        #         if err < tol:
        #             return u
        #         else:
        #             nonlocal failure_reason
        #             failure_reason = f"{method_name} found angle but error too large ({err:.3e})"
        #             return nan


        #     start_guesses = [u0_guess, 0.0, 0.05, -0.05, 0.1, -0.1,.2,-.2,1,-1,2,-2]
        #     for guess in start_guesses:
        #         try:
        #             u_sol = newton(func, x0=guess, tol=1e-12, maxiter=50)
                
        #             checked = check_solution(u_sol, f"Newton start={guess}")
        #             if not isnan(checked):
                        
        #                 u_res= checked
        #                 y_res= func(checked) + self.h_obj
        #                 break
        #         except RuntimeError:
        #             continue
            
        #     result= _trace( u0=u_res, y_start=y_start, intersection_method=intersection_method, tol=tol)

        #     return result.U_rad.values.tolist()[-1]
            


        # def _stop_key_and_index(self):
        #     Surfaces = self.Surfaces.copy(deep=True)
        #     keys = Surfaces.keys().to_list()
        #     stop_keys = [k for k in keys if getattr(Surfaces[k], 'ast', False)]
        #     if not stop_keys:
        #         raise RuntimeError("No aperture stop flagged. Set one (self.set_Ast(...)) first.")
        #     stop_key = stop_keys[0]
        #     stop_idx = keys.index(stop_key)
        #     return stop_key, stop_idx, keys

        # def _trace_until_stop(self, y_ent, u_ent, stop_idx, intersection_method='distance', tol=1e-10):
        #     """
        #     Returns (y_at_stop, u_at_stop, vignetted_flag)
        #     Uses your existing _solver to march from 'ent' to the stop surface.
        #     """
        #     Surfaces = self.Surfaces.copy(deep=True)
        #     keys = Surfaces.keys().to_list()

        #     # Collect the running state the same way Solve_System does
        #     y = y_ent
        #     u = u_ent
        #     dz = 0.0
        #     vignetted = False

        #     # Start at the first real surface after 'obj'
        #     for i, surf in enumerate(keys):
        #         if i == 0:  # 'obj'
        #             continue
        #         if surf == 'ims':
        #             break

        #         surf_crnt = Surfaces[surf]
        #         surf_nxt_key = keys[i+1]
        #         surf_nxt = Surfaces[surf_nxt_key]

        #         z = surf_crnt.thickness
        #         z_act = z - dz

        #         # Step once
        #         y1, u1, z_int, err = self._solver(y, u, surf_crnt, surf_nxt, z_act,
        #                                         intersection_method=intersection_method, tol=tol)
        #         # vignetting check at this surface (optional but helpful)
        #         try:
        #             ap = float(getattr(surf_nxt, 'aperture', float('inf')))
        #             if abs(y1) > ap:
        #                 vignetted = True
        #         except:
        #             pass

        #         # If next surface IS the stop, return its hit height
        #         if i+1 == stop_idx:
        #             return y1, u1, vignetted

        #         # Otherwise continue marching
        #         dz = z_int - z_act
        #         y, u = y1, u1

        #     # Should never get here if stop_idx valid
        #     raise RuntimeError("Stop surface index not encountered while tracing.")

        # def _chief_objective(self, y_ent, stop_idx, object_at_infinity=False, field_angle_rad=None):
        #     """
        #     Returns y_at_stop for a given entrance height y_ent.
        #     Chooses u_ent according to your conventions.
        #     """
        #     if object_at_infinity:
        #         if field_angle_rad is None:
        #             # Use whatever you store for field angle at infinity
        #             u_ent = getattr(self, 'field_angle_rad', 0.0)
        #         else:
        #             u_ent = field_angle_rad
        #     else:
        #         # Finite object distance: your exact default convention
        #         u_ent = arctan((y_ent - self.h_obj) / self.Obj2ENT)

        #     y_stop, _, _ = self._trace_until_stop(y_ent, u_ent, stop_idx)
        #     return y_stop, u_ent

        # def find_chief_ray(self, object_at_infinity=False, field_angle_rad=None,
        #                 y_guess=None, tol=1e-8, max_iter=20, intersection_method='distance'):
        #     """
        #     Solve for (y_ent*, u_ent*) such that y_stop = 0 (chief ray).
        #     Returns (y_ent, u_ent).
        #     """
        #     stop_key, stop_idx, keys = self._stop_key_and_index()

        #     # Initial guesses for the entrance height
        #     if y_guess is None:
        #         # Sensible automatic guesses:
        #         #  - use the object height projected to 'ent' plane if finite object
        #         #  - or 0 and small offset if infinity
        #         if object_at_infinity:
        #             y1 = 0.0
        #             y2 = 0.5 * getattr(self, 'AST_size', 1.0)  # small lateral offset
        #         else:
        #             # project a straight line from object to stop z ignoring refraction:
        #             # y_ent ≈ h_obj + u_free * Obj2ENT, where u_free aims to stop center
        #             # u_free ~ (0 - h_obj) / (z_stop - z_obj), but we can just bracket
        #             y1 = self.h_obj
        #             y2 = self.h_obj + 0.25 * getattr(self, 'AST_size', 1.0)
        #     else:
        #         y1 = float(y_guess)
        #         y2 = y1 + 0.1 * max(1.0, abs(y1))

        #     # Evaluate objective at two points
        #     f1, u1 = self._chief_objective(y1, stop_idx, object_at_infinity, field_angle_rad)
        #     f2, u2 = self._chief_objective(y2, stop_idx, object_at_infinity, field_angle_rad)

        #     # Secant iterations on y_ent
        #     it = 0
        #     while it < max_iter and abs(f2) > tol:
        #         if f2 == f1:
        #             # perturb to avoid divide-by-zero
        #             y2 += 1e-6 if y2 == y1 else 0.0
        #             f2, u2 = self._chief_objective(y2, stop_idx, object_at_infinity, field_angle_rad)

        #         y_new = y2 - f2 * (y2 - y1) / (f2 - f1)

        #         # Slide the bracket
        #         y1, f1 = y2, f2
        #         y2 = y_new
        #         f2, u2 = self._chief_objective(y2, stop_idx, object_at_infinity, field_angle_rad)
        #         it += 1

        #     if abs(f2) > tol:
        #         raise RuntimeError("Chief-ray secant solver did not converge.")

        #     # Return entrance (height, angle) for the chief ray
        #     return y2, u2






        # def _optimizer(self,x0 , surf2optimize, y_target, surf_target, parameter='thickness', optimizer='rms', intersection_method='distance', tol=1e-10):

        #     """
        #     I dont know what to write here

        #     """
        #     Surfaces = self.Surfaces
        #     surfaces_keys = list(Surfaces.keys())



        #     index = None
        #     for i, sk in enumerate(surfaces_keys):
        #         if sk==surf2optimize.lower():
        #             index = i
        #             break
        #     if index is None:
        #         raise ValueError(f"{surf2optimize} surface was not found.")
            

        #     index_target = None
        #     for i, sk in enumerate(surfaces_keys):
        #         if sk==surf_target.lower():
        #             index_target = i
        #             if sk=='ims':
        #                 ims_prot=False
        #             else:
        #                 ims_prot=True
        #             break
        #     if index_target is None:
        #         raise ValueError(f"{surf_target} surface was not found.")

        #     if optimizer.lower() =='rms':
                
        #         if parameter.lower() == 'thickness':
        #             Surfaces[surfaces_keys[index]].thickness=x0

        #         elif parameter.lower() == 'radius':
        #             Surfaces[surfaces_keys[index]].radius=x0

        #         Ray_Result= self._default_solver_systm(n_rays=21, intersection_method=intersection_method, tol=tol, Surfaces=Surfaces)  

        #         y_mm=[df['Y_mm'].iloc[index_target-1] for df in Ray_Result]
        #         return sqrt(mean((array(y_mm) - y_target)**2))




        # def Opimizer(self,surf2optimize, y_target, surf_target, intersection_method='distance', parameter='thickness', optimizer='rms', tol=1e-10, x_guess=0.0, verbose=True):
        #     """
        #     Ultra-robust chief/marginal ray angle finder with debug logging.
        #     - Multi-start Newton
        #     - Trust-region least squares
        #     - Direct minimization
        #     - Auto-expanding bracket + Brent's method
        #     - Safety limit on angles
        #     - Returns NaN if error too large
        #     - Logs failure reasons if verbose=True
        #     """
        #     func = lambda x0: self._optimizer(x0 , surf2optimize, y_target, surf_target, parameter, optimizer, intersection_method, tol)

        
        #     failure_reason = None

        #     def check_solution(u, method_name):
        #         err = abs(func(u))
        #         if err < tol:
        #             return u
        #         else:
        #             nonlocal failure_reason
        #             failure_reason = f"{method_name} found angle but error too large ({err:.3e})"
        #             return nan

        #     # ---- 1. Multi-start Newton ----
        #     start_guesses = [x_guess]
        #     for guess in start_guesses:
        #         try:
        #             u_sol = newton(func, x0=guess, tol=1e-12, maxiter=50)
        #             checked = check_solution(u_sol, f"Newton start={guess}")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #         except RuntimeError:
        #             continue

        #     # # ---- 2. Trust-region reflective least squares ----
        #     # try:
        #     #     res = least_squares(lambda x: func(x[0]), x0=[x_guess], method='trf', ftol=1e-12, xtol=1e-12)
                
        #     #     checked = check_solution(res.x[0], "Least squares")
        #     #     if not isnan(checked):
        #     #         return checked, func(checked) + y_target
        #     # except Exception:
        #     #     pass

        #     # # ---- 3. Direct minimization of absolute error ----
        #     # try:
        #     #     min_res = minimize(lambda x: abs(func(x[0])), x0=[x_guess], method='Powell', tol=1e-12)
                
        #     #     checked = check_solution(min_res.x[0], "Minimize")
        #     #     if not isnan(checked):
        #     #         return checked, func(checked) + y_target
        #     # except Exception:
        #     #     pass





        # def _ray_height_at_ast(self, u0, y_target, intersection_method='distance', tol=1e-10):
        #     """
        #     Trace a single ray from object to AST, return difference between ray height at AST and target.
        #     """
        #     Surfaces = self.Surfaces.copy(deep=True)
        #     surfaces_keys = list(Surfaces.keys())

        #     # Starting point at obj surface
            
        #     y_init =  self.transfer(self.h_obj,u0, Surfaces.obj.thickness)
        #     Y_inter= [y_init]
        #     U= [u0]
        #     ERR= [0]
        #     Z_int= [0]
        #     Z= [0]
        #     dZ= [0]
        #     Z_act= [0]
        #     dz=0

        #     # Find AST index
        #     ast_index = None
        #     for i, sk in enumerate(surfaces_keys):
        #         if Surfaces[sk].ast:
        #             ast_index = i
        #             if sk=='ims':
        #                 ims_prot=False
        #             else:
        #                 ims_prot=True
        #             break
        #     if ast_index is None:
        #         raise ValueError("No AST surface found.")

        #     # Trace ray until AST surface
        #     for i, surf in enumerate(surfaces_keys):
        #         if i==0 or surf=='ims': continue 
        #         surf_crnt = Surfaces[surfaces_keys[i]]
        #         surf_nxt  = Surfaces[surfaces_keys[i+1]]
        #         if ims_prot:
        #             if surfaces_keys[i]==surfaces_keys[ast_index+1]:
        #                 break

        #         y0= Y_inter[i-1]
        #         u0= U[i-1]
        #         z= surf_crnt.thickness
        #         z_act= z-dz
        #         y1,u1, z_int,err= self._solver(y0,u0,surf_crnt,surf_nxt,z_act, intersection_method=intersection_method, tol=tol)
        #         dz= z_int-z_act
        #         Y_inter.append(y1)
        #         U.append(u1)
        #         ERR.append(err)
        #         Z_int.append(z_int)
        #         Z.append(z)
        #         Z_act.append(z_act)
        #         dZ.append(dz)

        #     result= DataFrame({'Z_mm':Z, 'Z_int_mm':Z_int,'Z_act_mm':Z_act,'dZ_mm':dZ,'Y_mm':Y_inter, 'U_rad':U, 'ERR':ERR})
            
        #     z_int_mm= cumsum(Z_int)
        #     y_mm=Y_inter

        #     z_int_mm[-1]=round(z_int_mm[-1], 4)
        #     func= interp1d(z_int_mm, y_mm)

        #     z_mm= cumsum(result.Z_mm.tolist())
        #     # z_mm= cumsum(RT.Surfaces.loc['thickness', :].values)[:-1].tolist()
        #     ASTs=self.Surfaces.loc['ast',:][1:].tolist()
        #     # print(ASTs)
        #     for (ast,zap) in zip(ASTs,z_mm):
                
        #         if ast:
        #             # print('I#am herrrrrr', zap)
        #             z_ap= zap
        #     # print(z_int_mm, y_mm, z_ap)
        #     y_ap= func(z_ap)
    
        #     return (y_ap - y_target)  # zero when ray hits exactly AST edge



        # def _chif_marg_ang_finder(self, y_target, intersection_method='distance', tol=1e-10, u_guess=0.0, verbose=True):
        #     """
        #     Ultra-robust chief/marginal ray angle finder with debug logging.
        #     - Multi-start Newton
        #     - Trust-region least squares
        #     - Direct minimization
        #     - Auto-expanding bracket + Brent's method
        #     - Safety limit on angles
        #     - Returns NaN if error too large
        #     - Logs failure reasons if verbose=True
        #     """
        #     func = lambda u: self._ray_height_at_ast(u, y_target, intersection_method, tol)

        #     ANGLE_LIMIT = 1.5  # rad (~86°)
        #     failure_reason = None

        #     def check_solution(u, method_name):
        #         err = abs(func(u))
        #         if err < tol:
        #             return u
        #         else:
        #             nonlocal failure_reason
        #             failure_reason = f"{method_name} found angle but error too large ({err:.3e})"
        #             return nan

        #     # ---- 1. Multi-start Newton ----
        #     start_guesses = [u_guess, 0.0, 0.05, -0.05, 0.1, -0.1]
        #     for guess in start_guesses:
        #         try:
        #             u_sol = newton(func, x0=guess, tol=1e-12, maxiter=50)
        #             if abs(u_sol) < ANGLE_LIMIT:
        #                 checked = check_solution(u_sol, f"Newton start={guess}")
        #                 if not isnan(checked):
        #                     return checked, func(checked) + y_target
        #         except RuntimeError:
        #             continue

        #     # ---- 2. Trust-region reflective least squares ----
        #     try:
        #         res = least_squares(lambda u: func(u[0]), x0=[u_guess], method='trf', ftol=1e-12, xtol=1e-12)
        #         if res.success and abs(res.x[0]) < ANGLE_LIMIT:
        #             checked = check_solution(res.x[0], "Least squares")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #     except Exception:
        #         pass

        #     # ---- 3. Direct minimization of absolute error ----
        #     try:
        #         min_res = minimize(lambda u: abs(func(u[0])), x0=[u_guess], method='Powell', tol=1e-12)
        #         if min_res.success and abs(min_res.x[0]) < ANGLE_LIMIT:
        #             checked = check_solution(min_res.x[0], "Minimize")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #     except Exception:
        #         pass

        #     # ---- 4. Auto-expand bracket for Brent ----
        #     u_low, u_high = -0.05, 0.05
        #     f_low, f_high = func(u_low), func(u_high)
        #     expand_factor = 1.5
        #     max_expansions = 50

        #     for _ in range(max_expansions):
        #         if f_low * f_high <= 0:
        #             break
        #         if abs(u_low) < ANGLE_LIMIT:
        #             u_low *= expand_factor
        #         if abs(u_high) < ANGLE_LIMIT:
        #             u_high *= expand_factor
        #         f_low, f_high = func(u_low), func(u_high)
        #     else:
        #         failure_reason = "No bracket found for Brent"
        #         if verbose:
        #             print(f"⚠️ {failure_reason}")
        #         return nan, nan

        #     try:
        #         u_sol = brentq(func, u_low, u_high, xtol=1e-12, maxiter=500)
        #         if abs(u_sol) < ANGLE_LIMIT:
        #             checked = check_solution(u_sol, "Brent")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #     except Exception:
        #         failure_reason = "Brent failed"
        #         if verbose:
        #             print(f"⚠️ {failure_reason}")
        #         return nan, nan

        #     if verbose:
        #         print(f"⚠️ No solution found. Last reason: {failure_reason}")
        #     return nan, nan





        # def _ray_height_at_ast_infinity(self, y0, y_target, intersection_method='distance', tol=1e-10):
        #     """
        #     Trace a single ray from object to AST, return difference between ray height at AST and target.
        #     """
        #     Surfaces = self.Surfaces.copy(deep=True)
        #     surfaces_keys = list(Surfaces.keys())

        #     # Starting point at obj surface
            
        #     u_init =  arctan((y0-self.h_obj)/self.Obj2ENT)
        #     Y_inter= [y0]
        #     U= [u_init]
        #     ERR= [0]
        #     Z_int= [0]
        #     Z= [0]
        #     dZ= [0]
        #     Z_act= [0]
        #     dz=0

        #     # Find AST index
        #     ast_index = None
        #     for i, sk in enumerate(surfaces_keys):
        #         if Surfaces[sk].ast:
        #             ast_index = i
        #             if sk=='ims':
        #                 ims_prot=False
        #             else:
        #                 ims_prot=True
        #             break
        #     if ast_index is None:
        #         raise ValueError("No AST surface found.")

        #     # Trace ray until AST surface
        #     for i, surf in enumerate(surfaces_keys):
        #         if i==0 or surf=='ims': continue 
        #         surf_crnt = Surfaces[surfaces_keys[i]]
        #         surf_nxt  = Surfaces[surfaces_keys[i+1]]
        #         if ims_prot:
        #             if surfaces_keys[i]==surfaces_keys[ast_index+1]:
        #                 break

        #         y0= Y_inter[i-1]
        #         u0= U[i-1]
        #         z= surf_crnt.thickness
        #         z_act= z-dz
        #         y1,u1, z_int,err= self._solver(y0,u0,surf_crnt,surf_nxt,z_act, intersection_method=intersection_method, tol=tol)
        #         dz= z_int-z_act
        #         Y_inter.append(y1)
        #         U.append(u1)
        #         ERR.append(err)
        #         Z_int.append(z_int)
        #         Z.append(z)
        #         Z_act.append(z_act)
        #         dZ.append(dz)

        #     result= DataFrame({'Z_mm':Z, 'Z_int_mm':Z_int,'Z_act_mm':Z_act,'dZ_mm':dZ,'Y_mm':Y_inter, 'U_rad':U, 'ERR':ERR})
            
        #     z_int_mm= cumsum(Z_int)
        #     y_mm=Y_inter
        #     z_int_mm[-1]=round(z_int_mm[-1], 4)
        #     func= interp1d(z_int_mm, y_mm)


        #     z_mm= cumsum(result.Z_mm.tolist())
        #     # z_mm= cumsum(RT.Surfaces.loc['thickness', :].values)[:-1].tolist()
        #     ASTs=self.Surfaces.loc['ast',:][1:].tolist()
        #     # print(ASTs)
        #     for (ast,zap) in zip(ASTs,z_mm):
                
        #         if ast:
        #             # print('I#am herrrrrr', zap)
        #             z_ap= zap
        #     y_ap= func(z_ap)
        #     return (y_ap - y_target)  # zero when ray hits exactly AST edge




        # def _chif_marg_ang_finder_infinity(self, y_target, intersection_method='distance', tol=1e-10, y_guess=5, verbose=True):
        #     """
        #     Ultra-robust chief/marginal ray angle finder with debug logging.
        #     - Multi-start Newton
        #     - Trust-region least squares
        #     - Direct minimization
        #     - Auto-expanding bracket + Brent's method
        #     - Safety limit on angles
        #     - Returns NaN if error too large
        #     - Logs failure reasons if verbose=True
        #     """
        #     func = lambda y: self._ray_height_at_ast_infinity(y, y_target, intersection_method, tol)

        #     HEIGHT_LIMIT = self.Surfaces.loc['radius', self.Surfaces.keys().tolist()[1]]+10
        #     failure_reason = None

        #     def check_solution(y, method_name):
        #         err = abs(func(y))
        #         if err < tol:
        #             return y
        #         else:
        #             nonlocal failure_reason
        #             failure_reason = f"{method_name} found heeight but error too large ({err:.3e})"
        #             return nan

        #     # ---- 1. Multi-start Newton ----
        #     start_guesses = [y_guess, 2,4,6,8,10, -2,-4,-6,-8,-10]
        #     for guess in start_guesses:
        #         try:
        #             u_sol = newton(func, x0=guess, tol=1e-12, maxiter=50)
        #             if abs(u_sol) < HEIGHT_LIMIT:
        #                 checked = check_solution(u_sol, f"Newton start={guess}")
        #                 if not isnan(checked):
        #                     return checked, func(checked) + y_target
        #         except RuntimeError:
        #             continue

        #     # ---- 2. Trust-region reflective least squares ----
        #     try:
        #         res = least_squares(lambda y: func(y[0]), x0=[y_guess], method='trf', ftol=1e-12, xtol=1e-12)
        #         if res.success and abs(res.x[0]) < HEIGHT_LIMIT:
        #             checked = check_solution(res.x[0], "Least squares")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #     except Exception:
        #         pass

        #     # ---- 3. Direct minimization of absolute error ----
        #     try:
        #         min_res = minimize(lambda y: abs(func(y[0])), x0=[y_guess], method='Powell', tol=1e-12)
        #         if min_res.success and abs(min_res.x[0]) < HEIGHT_LIMIT:
        #             checked = check_solution(min_res.x[0], "Minimize")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #     except Exception:
        #         pass

        #     # ---- 4. Auto-expand bracket for Brent ----
        #     y_low, y_high = -0.05, 0.05
        #     f_low, f_high = func(y_low), func(y_high)
        #     expand_factor = 1.5
        #     max_expansions = 50

        #     for _ in range(max_expansions):
        #         if f_low * f_high <= 0:
        #             break
        #         if abs(y_low) < HEIGHT_LIMIT:
        #             y_low *= expand_factor
        #         if abs(y_high) < HEIGHT_LIMIT:
        #             u_high *= expand_factor
        #         f_low, f_high = func(y_low), func(y_high)
        #     else:
        #         failure_reason = "No bracket found for Brent"
        #         if verbose:
        #             print(f"⚠️ {failure_reason}")
        #         return nan, nan

        #     try:
        #         u_sol = brentq(func, y_low, y_high, xtol=1e-12, maxiter=500)
        #         if abs(u_sol) < HEIGHT_LIMIT:
        #             checked = check_solution(u_sol, "Brent")
        #             if not isnan(checked):
        #                 return checked, func(checked) + y_target
        #     except Exception:
        #         failure_reason = "Brent failed"
        #         if verbose:
        #             print(f"⚠️ {failure_reason}")
        #         return nan, nan

        #     if verbose:
        #         print(f"⚠️ No solution found. Last reason: {failure_reason}")
        #     return nan, nan



        # def _chief_marginal_solver_systm(self, y_target, intersection_method='distance', tol=1e-10, u=None, y=None):
        #     if self.Obj2ENT<1e6:
        #         if u ==None:
        #             u,y_at_ast=self._chif_marg_ang_finder(y_target, intersection_method=intersection_method, tol=tol)
            
        #     else:
        #         if y==None:
        #             y,y_at_ast=self._chif_marg_ang_finder_infinity(y_target, intersection_method=intersection_method, tol=tol)
                
        #     if self.Obj2ENT<1e6:
        #         y_init = tan(u)*self.Obj2ENT+self.h_obj
        #         u_init = u
        #     else:
        #         y_init = y
        #         u_init = arctan((y-self.h_obj)/self.Obj2ENT)
        #     self.y_chief_marg= y_init
        #     Surfaces = self.Surfaces.copy(deep=True)
        #     surfaces_keys = Surfaces.keys().to_list()

        #     if not any([Surfaces[sk].ast for sk in surfaces_keys ]):
        #             self.set_Ast(self.y0, surface_index=1)
        #             ast_index=1
        #     else:
        #         # Determine where the AST actually is
        #         ast_index = None
        #         for i, sk in enumerate(surfaces_keys):
        #             if Surfaces[sk].ast:
        #                 ast_index = i
        #                 break
        

        #     Rays_Result=[]

        #     self.Y_inter= [y_init]
        #     self.U= [u_init]
        #     self.ERR= [0]
        #     self.Z_int= [0]
        #     self.Z= [0]
        #     self.dZ= [0]
        #     self.Z_act= [0]

        #     dz=0
        #     for i, surf in enumerate(surfaces_keys):
        #         if i==0 or surf=='ims': continue 

        #         surf_crnt= Surfaces[surf]
        #         surf_nxt= Surfaces[surfaces_keys[i+1]]
        #         y0= self.Y_inter[i-1]
        #         u0= self.U[i-1]
        #         z= surf_crnt.thickness
        #         z_act= z-dz
        #         y1,u1, z_int,err= self._solver(y0,u0,surf_crnt,surf_nxt,z_act, intersection_method=intersection_method, tol=tol)
        #         dz= z_int-z_act
        #         self.Y_inter.append(y1)
        #         self.U.append(u1)
        #         self.ERR.append(err)
        #         self.Z_int.append(z_int)
        #         self.Z.append(z)
        #         self.Z_act.append(z_act)
        #         self.dZ.append(dz)

        #         Results= DataFrame({
        #             'Z_mm':self.Z, 'Z_int_mm':self.Z_int,'Z_act_mm':self.Z_act,'dZ_mm':self.dZ, 
        #             'Y_mm':self.Y_inter, 'U_rad':self.U, 'ERR':self.ERR
        #         })
        #         # Results.index=[s.upper() for s in surfaces_keys[1:]]
        #     Rays_Result.append(Results)
        #     return Rays_Result

