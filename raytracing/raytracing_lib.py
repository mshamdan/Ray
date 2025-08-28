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
        self.check_chief= True
        self.h_obj_arb=None


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

        # print(h_obj)
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
            Rays_Result= self._default_solver_systm(n_rays, intersection_method, tol, u_rad=u_rad, y_mm=y_mm,h_obj=h_obj, FOV_deg=FOV_deg)
            if h_obj==None and FOV_deg==None:
                self.h_obj_arb= self.h_obj
            
            elif h_obj==None and FOV_deg!=None:
                self.h_obj_arb= self.obj_height_from_FOV(FOV_deg)
            else:
                self.h_obj_arb= h_obj
            
            if u_rad!=None:
                self.y_arb = self.h_obj_arb+ (tan(u_rad)*self.Obj2ENT)
            else:
                self.y_arb = y_mm
            
            return Rays_Result


    def _ast_solved_check(self,marginal_ang_tol=1e-6, ST_Active:bool=False):
            # # print(1)
            if all([self.u_marg==None , self.y_marg==None]) or ST_Active!=self.active_stop:
                
                # # print(2)
                if self.Obj2ENT<1e4:
                    # # print(3)
                    self.AST_surface,u= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                    self.u_marg=u
                else:
                    # # print(4)
                    self.AST_surface,y= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                    self.y_marg=y
                # # print(5)
                self.Surfaces_copy=self.Surfaces.copy(deep=True)
                self.active_stop= ST_Active
                self.check_chief= True
                return True
            
            else:
                # # print(6)
                if ST_Active:
                    # # print(7)
                    if not  self.Surfaces.iloc[:10,:].equals(self.Surfaces_copy.iloc[:10,:]):
                        # # print(8)
                        if self.Obj2ENT<1e4:
                            # # print(9)
                            self.AST_surface,u= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.u_marg=u
                        else:
                            # # print(10)
                            self.AST_surface,y= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.y_marg=y
                        # # print(11)
                        self.Surfaces_copy=self.Surfaces.copy(deep=True)
                        self.active_stop= ST_Active
                        self.check_chief= True
                        return True


                else:
                    # # print(12)

                    if not self.Surfaces.iloc[:8,:].equals(self.Surfaces_copy.iloc[:8,:]):
                        # # print(13)
                        if self.Obj2ENT<1e4:
                            self.AST_surface,u= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.u_marg=u
                            # # print(14)
                        else:
                            # # print(15)
                            self.AST_surface,y= self.find_AST(tol=marginal_ang_tol,ST_Active=ST_Active)
                            self.y_marg=y
                        # # print(16)
                        self.Surfaces_copy=self.Surfaces.copy(deep=True)
                        self.active_stop= ST_Active
                        self.check_chief= True
                        return True
            # # print(17)
            return False


    def _chief_solved_check(self,marginal_ang_tol=1e-6, tol=1e-6, FOV_deg=None, ST_Active:bool=False,chief_guess=0):


        if self._ast_solved_check(ST_Active=ST_Active) or self.check_chief:
            # # print(11)
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0
            self.check_chief= False

        elif all([self.FOV_deg== None, self.h_obj_chief==None]) or self.y_chief==None or self.check_chief:
            # # print(22)
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0
            self.check_chief= False
        
        elif self.FOV_deg!=FOV_deg or self.check_chief:
            # # print(33)
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0
            self.check_chief= False
        
        elif all([self.FOV_deg==None , self.h_obj_chief!= self.h_obj]) or self.check_chief:
            # # print(44)
            y0, h_obj, err= self.find_max_height_chief( fov_deg=FOV_deg, marginal_ang_tol=marginal_ang_tol, tol=tol, ST_Active=ST_Active, chief_guess=chief_guess)
            self.FOV_deg= FOV_deg
            self.h_obj_chief=h_obj
            self.y_chief=y0
            self.check_chief= False
        

    def _plot_system(self,fig=None, ax=None, n_rays:int=3,ST_Active:bool=True,show_obj:bool=False, show_img:bool=True, 
                     lens_layout:bool=True, linestyle:str='-', ray_color:str='r',obj_col:str='orange', figsize=(10,4), 
                     Results=None, xscale:str='symlog',yscale:str='symlog',solver:str='default', intersection_method:str='distance', 
                     tol=1e-10,marginal_ang_tol=1e-6 , u_rad=None, y_mm=None, h_obj=None, FOV_deg=None, chief_guess=0):
        
        if lens_layout:
            fig,ax= self.Lens_Layout(fig, ax,show_obj, show_img,ST_Active,obj_col, figsize, xscale=xscale, yscale=yscale)
        elif fig==None or ax==None:
            fig,ax= subplots(figsize=figsize, tight_layout=True)

        Ray_Result=Results
        # # # print('Results', Results)
        if Results is None:
            Ray_Result= self.Solve_System(n_rays=n_rays,solver=solver, intersection_method=intersection_method,ST_Active=ST_Active,
                                          tol=tol, marginal_ang_tol=marginal_ang_tol, u_rad=u_rad, y_mm=y_mm, h_obj=h_obj, FOV_deg=FOV_deg,chief_guess=chief_guess)
        # # # print(Ray_Result[0].Y_mm.values)
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
                    # # # print(h_obj, y_init)
                    
                if solver.lower() in ['chief']:
                    n_rays=1
                    h_obj= self.h_obj_chief
                    y_0= self.y_chief

                rays= -linspace(-1,1, n_rays, endpoint=True)
                for ray in rays:

                    y_init= y_0*ray

                    if solver.lower() in ['arbitrary']:
                        y_init= self.y_arb*ray
                        h_obj= self.h_obj_arb

                    # # # print([0, z_int_mm[0]], [h_obj, y_init])
                    ax.plot([0, z_int_mm[0]], [h_obj, y_init], linestyle=linestyle,color=ray_color)
                    if solver.lower() in ['chief']:
                        ax.plot([0, z_int_mm[0]], [-h_obj, -y_init], linestyle=linestyle,color=ray_color)
                
###############################################
            if ST_Active and solver.lower() not in ['marginal', 'm', 'marg']:
                z_int_mm, y_mm,_= self._check_stop(result, self.Surfaces.copy(deep=True), dz,ST_Active=ST_Active)

            
            ax.plot(z_int_mm, y_mm, linestyle=linestyle,color=ray_color)
            if solver.lower() in ['chief']:
                ax.plot(z_int_mm, -y_mm, linestyle=linestyle,color=ray_color)
        
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


    def _check_stop(self, result, Surfaces, dz=0, ST_Active:bool=False):
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

        if ST_Active:
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
                if stop:
                    if abs(stop_size)>abs(ap):
                        if abs(ap)<abs(R):
                            y_comp= abs(ap)
                        else:
                            y_comp= abs(R)-1e-10
                    else:
                        y_comp= abs(stop_size)
                else:
                    if abs(ap)<abs(R):
                        y_comp= abs(ap)
                    else:
                        y_comp= abs(R)-1e-10


                if abs(y_ap)> y_comp:
                    # # # print(11111)
                    z_int_mm= [i for i in z_int_mm if i <=z_ap]
                    surface_res= [surf_keys[w] for w,i in enumerate(z_int_mm) if i <=z_ap]
                    len_diff= len(surf_keys)- len(z_int_mm)
                    if z_ap not in z_int_mm:
                        z_int_mm+=[z_ap]
                        surface_res+= [S.name]
                    # if len_diff>0:
                    #     for i in range(len_diff):
                    #         z_int_mm+=[nan]

                    y_mm= func(z_int_mm)
                    break 
            
            
            
            if len(Z_ap)==0:
                z_int_mm= [i for i,k in zip(z_int_mm, y_mm) if not isnan(k)]
                y_mm= [k for i,k in zip(z_int_mm, y_mm) if not isnan(k)]
                surface_res= [surf_keys[w] for w,i in enumerate(z_int_mm)]
        else:
            z_int_mm= [i for i,k in zip(z_int_mm, y_mm) if not isnan(k)]
            y_mm= [k for i,k in zip(z_int_mm, y_mm) if not isnan(k)]
            surface_res= [surf_keys[w] for w,i in enumerate(z_int_mm)]

        return z_int_mm, y_mm, surface_res
    

    def Check_Stop(self, result, Surfaces, dz=0, ST_Active:bool=False):
        surface_res= Surfaces.keys().values
        z_int_mm= cumsum(result.Z_int_mm.values)+dz
        y_mm=result.Y_mm.values
        U_rad=result.U_rad.values
        func= interp1d(z_int_mm, y_mm)

        z_mm= cumsum(result.Z_mm.tolist())+dz

        STs=Surfaces.loc['stop',:][1:].tolist()
        ST_sizes=Surfaces.loc['stop_size',:][1:].tolist()

        surf_keys=(Surfaces.keys().tolist())[1:]
        Z_ap= []
        Ss= []

        if ST_Active:
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
                if stop:
                    if abs(stop_size)>abs(ap):
                        if abs(ap)<abs(R):
                            y_comp= abs(ap)
                        else:
                            y_comp= abs(R)-1e-10
                    else:
                        y_comp= abs(stop_size)
                else:
                    if abs(ap)<abs(R):
                        y_comp= abs(ap)
                    else:
                        y_comp= abs(R)-1e-10


                if abs(y_ap)> y_comp:
                    surface_res= []
                    z_res= []
                    y_res= []
                    u_res= []


                    for (z_in, u,y, surf) in zip(z_int_mm, U_rad,y_mm ,surf_keys):
                        if z_in<z_ap:
                            surf_keys.append(surf)
                            z_res.append(z_in)
                            y_res.append(y)
                            u_res.append(u)
                        else:
                            surf_keys.append(surf)
                            z_res.append(z_in)
                            y_res.append(nan)
                            u_res.append(nan)
                    y_mm= y_res
                    U_rad= u_res
                    result.loc[:, "Y_mm"] = y_mm
                    result.loc[:, "U_rad"] = U_rad
                    break
                

        return result


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
                    # # # print(ASTs)
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
                        
                        
                        # # # print(z_ap)
                        yyy= abs(R)-1e-10
                        if ap<abs(R):
                            yyy= ap
                        if not any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                            if typ=='asph':
                                z_ap= self.sag_aspherical(yyy, R, k, As, z_ap)
                                # # # print(z_ap)

                            else:
                                z_ap= self.sag_spherical(yyy, R, z_ap)
                                # # # print(z_ap)

                    
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
                    # # # print(ASTs)
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
                        
                        
                        # # # print(z_ap)
                        yyy= abs(R)-1e-10
                        if ap<abs(R):
                            yyy= ap
                        if not any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                            if typ=='asph':
                                z_ap= self.sag_aspherical(yyy, R, k, As, z_ap)
                                # # # print(z_ap)

                            else:
                                z_ap= self.sag_spherical(yyy, R, z_ap)
                                # # # print(z_ap)

                    
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
            # # # print(1)
            
            max_angle, Result, Y_ap, S_ap,ERR_ap = self.find_max_angle(tol=tol,ST_Active=ST_Active)

            err= min(abs(abs(Surfaces.iloc[3,:].values[1:])-abs(Result[0].Y_mm.values)))

            Rays_Result= self._default_solver_systm(n_rays=1, u_rad=max_angle+1e-3,h_obj=0)
            if not ST_Active:
                # # # print(2)
                # # # print(Rays_Result[0].Y_mm.values)
                # # # print(Rays_Result[0].index.values)
                # idx_ast= where(isnan(Rays_Result[0].Y_mm.values))[0][0]
                # self.AST_surface= Rays_Result[0].index.values[int(idx_ast-1)]
                _, y_mm, surfaces= self._check_stop( Rays_Result[0], Surfaces,ST_Active=ST_Active)
                # # # print(y_mm)
                # # # print(surfaces)

                # idx_ast= int([i for i, y_nan in enumerate(isnan(y_mm)) if y_nan][0]-1)
                # self.AST_surface= surfaces[int(idx_ast)]
                surfaces= list(surfaces)
                try:
                    surfaces.remove('obj')
                except: pass
                surfaces= array(surfaces)

                self.AST_surface= [sur for i,(sur, yy_nan) in enumerate(zip(surfaces, isnan(y_mm))) if not yy_nan][-1]
                # self.AST_surface= surfaces[-1]


            else:
                # # # print(3)
                _, y_mm, surfaces= self._check_stop( Rays_Result[0], Surfaces,ST_Active=ST_Active)

                surfaces= list(surfaces)
                try:
                    surfaces.remove('obj')
                except: pass
                surfaces= array(surfaces)
                # # # print(y_mm)
                # # # print(surfaces)
                # # # print([sur for i,(sur, yy_nan) in enumerate(zip(surfaces, isnan(y_mm))) if not yy_nan])

                # idx_ast= int([i for i, y_nan in enumerate(isnan(y_mm)) if y_nan][0]-1)
                # # # print(idx_ast)
                self.AST_surface= [sur for i,(sur, yy_nan) in enumerate(zip(surfaces, isnan(y_mm))) if not yy_nan][-1]
                # self.AST_surface= surfaces[-1]
            
            if self.AST_surface=='ent':
                # # # print(4)
                if not Surfaces.ent.stop:
                    # # # print(5)
                    self.AST_surface='surf2'
                else:
                    # # # print(6)
                    Rays_Result= self._default_solver_systm(n_rays=1, u_rad=max_angle,h_obj=0)
                    y_ent= Rays_Result[0].Y_mm.values[0]
                    y_surf2= Rays_Result[0].Y_mm.values[1]
                    err_ent=abs(abs(y_ent)-abs(Surfaces.ent.stop_size))
                    
                    if Surfaces.surf2.stop:
                        # # # print(7)
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.stop_size))
                    else:
                        # # # print(8)
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.aperture))
                    
                    if err_ent>err_surf2:
                        # # # print(9)
                        self.AST_surface='surf2'

            # # # print(10)
            return self.AST_surface, max_angle
        
        else:
            # # # print(11)
            max_height, Result, Y_ap, S_ap, ERR_ap   = self.find_max_height_infinity(tol=tol,ST_Active=ST_Active)
            
            err= min(abs(abs(Surfaces.iloc[3,:].values[1:])-abs(Result[0].Y_mm.values)))
  
            Rays_Result= self._default_solver_systm(n_rays=1, y_mm=max_height+1e-3,h_obj=0)
            if not ST_Active:
                # # # print(12)
                idx_ast= where(isnan(Rays_Result[0].Y_mm))[0][0]
                self.AST_surface= Rays_Result[0].index.values[int(idx_ast-1)]

            else:
                # # # print(13)
                _, y_mm, surfaces= self._check_stop( Rays_Result[0], Surfaces,ST_Active=ST_Active)
                self.AST_surface= surfaces[-1]


            if self.AST_surface=='ent':
                # # # print(14)
                if not Surfaces.ent.stop:
                    # # # print(15)
                    self.AST_surface='surf2'
                else:
                    # # # print(16)
                    Rays_Result= self._default_solver_systm(n_rays=1, y_mm=max_height,h_obj=0)
                    y_ent= Rays_Result[0].Y_mm.values[0]
                    y_surf2= Rays_Result[0].Y_mm.values[1]
                    err_ent=abs(abs(y_ent)-abs(Surfaces.ent.stop_size))
                    
                    if Surfaces.surf2.stop:
                        # # # print(17)
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.stop_size))
                    else:
                        # # # print(18)
                        err_surf2=abs(abs(y_surf2)-abs(Surfaces.surf2.aperture))
                    
                    if err_ent>err_surf2:
                        # # # print(19)
                        self.AST_surface='surf2'

        
            # # # print(20)
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
            warn(f"⚠️ Warning: Requested FOV={fov_deg:.2f}° exceeds system limit {max_fov:.2f}°", UserWarning)

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
        # print(1, self.AST_surface)
        self._ast_solved_check(marginal_ang_tol=marginal_ang_tol, ST_Active=ST_Active)
        # print(2, self.AST_surface)
        y_max = self.Surfaces.surf2.aperture
        h_obj= self.h_obj
        if fov_deg!=None:
            h_obj = self.obj_height_from_FOV(FOV_deg=fov_deg)
        

        Surfaces = self.Surfaces.copy(deep=True)
        surf_keys = Surfaces.keys().values[1:]  # skip obj
        ast_idx = where(surf_keys == self.AST_surface)[0][0]

        # # # print(ast_idx)
        if ast_idx ==0:
            return 0, h_obj, 0

        if ast_idx ==1:
            u= arctan((0-h_obj)/(self.Obj2ENT+Surfaces.ent.thickness))
            # # # print(Surfaces.ent.thickness*tan(abs(u)))
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
                # # # print(y0, y_ast, h_obj)
                return y_ast**2  # squared error → smooth near 0
            except Exception:
                return 1e6  # penalty on failure


        func = lambda y: ERR(y[0], h_obj=h_obj)

        bnds = [(-y_max, y_max)]


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


    def Gaussian_fiber_solver(self, n_rays:int, MFD_um=None, NA=None, r_core_um=None,method="deterministic",seed=None,
                              intersection_method='distance', tol=1e-10):
        self.h_obj_fiber, U_rad, self.weight_fiber= self.gaussian_fiber(n_rays=n_rays, MFD_um=MFD_um, NA=NA, r_core_um= r_core_um, method=method, seed=seed)
        self.y_fiber= []
        Rays_Result= []
        for (h_obj, u_rad) in zip(self.h_obj_fiber, U_rad):
            rays_result= self._default_solver_systm(1, intersection_method, tol, u_rad=u_rad, h_obj=h_obj)
            y_mm = h_obj+ (tan(u_rad)*self.Obj2ENT)
            Rays_Result.append(rays_result[0])
            self.y_fiber.append(y_mm)        
        return Rays_Result


    def Gaussian_Rays_Solver(self, n_rays, w0_mm, z_mm=0.0, truncate_sigma=3.0, intersection_method='distance', method="deterministic", seed=None, tol=1e-10):    
        
        self.h_obj_gaussian, U_rad, self.weight_gaussian= self.gaussian_beam_rays(n_rays=n_rays,w0_mm=w0_mm, z_mm=z_mm, truncate_sigma=truncate_sigma, method=method, seed=seed)

        self.y_gaussian= []
        Rays_Result= []
        for (h_obj, u_rad) in zip(self.h_obj_gaussian, U_rad):
            rays_result= self._default_solver_systm(1, intersection_method, tol, u_rad=u_rad, h_obj=h_obj)
            y_mm = h_obj+ (tan(u_rad)*self.Obj2ENT)
            Rays_Result.append(rays_result[0])
            self.y_gaussian.append(y_mm)

        return Rays_Result


    def plot_Gaussian_System(self,fig=None, ax=None, n_rays:int=3,ST_Active:bool=True,show_obj:bool=False, show_img:bool=True, 
                    lens_layout:bool=True, linestyle:str='-', ray_color:str='r',obj_col:str='orange', figsize=(10,4), 
                    xscale:str='symlog',yscale:str='symlog',solver:str='fiber', intersection_method:str='distance', 
                    tol=1e-10, w0_mm:float=None, truncate_sigma_gaussian=1,MFD_um=None, NA=None, r_core_um=None, method="deterministic", seed=None):
        
        if lens_layout:
            fig,ax= self.Lens_Layout(fig, ax,show_obj, show_img,ST_Active,obj_col, figsize, xscale=xscale, yscale=yscale)
        elif fig==None or ax==None:
            fig,ax= subplots(figsize=figsize, tight_layout=True)


        if solver.lower() in ['fib', 'fiber', 'single-mode fiber', 'smf', 'multi-mode fiber', 'mmf']:
            if NA==None and MFD_um==None: pass
            else:
                Ray_Result= self.Gaussian_fiber_solver(n_rays=n_rays, MFD_um=MFD_um, NA=NA, r_core_um=r_core_um, method=method, seed=seed, intersection_method=intersection_method, tol=tol)

            H_obj= self.h_obj_fiber
            Y_ent= self.y_fiber

        if solver.lower() in ['gaussian', 'gauss', 'gauss ray']:
            if w0_mm==None: pass

            else:
                Ray_Result= self.Gaussian_Rays_Solver(n_rays=n_rays, w0_mm=w0_mm, z_mm=0,truncate_sigma= truncate_sigma_gaussian, method=method, seed=seed, tol=tol)
        
            H_obj= self.h_obj_gaussian
            Y_ent= self.y_gaussian



        dz= self.Obj2ENT
        if not show_obj:
            dz=0
        

        for result, h_obj, y_ent in zip(Ray_Result, H_obj, Y_ent):
            z_int_mm= cumsum(result.Z_int_mm.values)+dz
            y_mm=result.Y_mm.values
            if show_obj:
                h_obj= h_obj
                y_init= y_ent

                ax.plot([0, z_int_mm[0]], [h_obj, y_init], linestyle=linestyle,color=ray_color)

                
###############################################
            if ST_Active:
                z_int_mm, y_mm,_= self._check_stop(result, self.Surfaces.copy(deep=True), dz,ST_Active=ST_Active)

            # print(result)
            ax.plot(z_int_mm, y_mm, linestyle=linestyle,color=ray_color)
            if solver.lower() in ['chief']:
                ax.plot(z_int_mm, -y_mm, linestyle=linestyle,color=ray_color)
        
        return fig, ax, Ray_Result






    # def parameter_optimizer(self,surfac1_idx:int, surface2_idx:int, parameter:str, y_target:str, solver:str='default', ST_Active:bool=False): 
        
    #     if parameter
        
        
    #     Surfaces= self.Surfaces.copy(deep=True)