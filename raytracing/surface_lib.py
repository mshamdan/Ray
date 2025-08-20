from numpy import round, ones, argsort, cumsum, rad2deg, deg2rad, sqrt, inf,errstate, where,sign, nan, tan, array, dot, arccos, degrees,arctan, nan, full, isnan,vstack,linspace, isfinite,arcsin, argmin,random,linalg,clip, cos, sin
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre
from scipy.special import factorial
from scipy.constants import speed_of_light as c_light
import pandas as pd
from matplotlib.pyplot import subplots,show,grid
from .zmxreader_lib import ZMXreader

pd.set_option('display.float_format', '{:.1e}'.format)

class Surface(ZMXreader):
    """
    Represents a surface in the optical system.
    """

    def __init__(self, Ent_radius:float=0, Ent_material:str='air', Obj_2_Ent_tickness:float= 1e20, Ent_thickness:float=0, Ent_aperture:float=1e20, Ent_surface_type:str='sph', Ent_conic:float=0,Ent_A_list:list=[]):
        
        ZMXreader.__init__(self)

        if Ent_surface_type.lower() in ['sphere', 'spheric', 'spherical', 'sph']:
            Ent_surface_type='sph'
        
        elif Ent_surface_type.lower() in ['asphere', 'aspheric', 'aspherical', 'asph']:
            Ent_surface_type='asph'
        
        else:
            raise ValueError(f"Suface Type{Ent_surface_type} is not valid, it must be eihter sph or asph")
        
        self.object_surface={'surface_name': 'obj',
                             'radius': 0,
                             'thickness': Obj_2_Ent_tickness,
                             'aperture': 1e10, 
                             'material': 'air', 
                             'type': 'sph',
                             'conic': 0,
                             'A_coefficient': [],
                             'stop':False,
                             'stop_size':1e10,
                             'ast':False}
        
        self.Ent_surface={'surface_name': 'ent',
                        'radius': Ent_radius,
                        'thickness': Ent_thickness,
                        'aperture': Ent_aperture, 
                        'material': Ent_material, 
                        'type': Ent_surface_type,
                        'conic': Ent_conic,
                        'A_coefficient': Ent_A_list,
                        'stop':False,
                        'stop_size':Ent_aperture,
                        'ast':False}   

        self.image_surface={'surface_name':'ims',
                            'radius': 0,
                            'thickness': 0,
                            'aperture': 1e10, 
                            'material': 'air', 
                            'type': "sph",
                            'conic': 0,
                            'A_coefficient': [],
                            'stop':False,
                            'stop_size':1e10,
                            'ast':False}
        
        
        
        self.Surfaces= pd.DataFrame({'obj':self.object_surface, 'ent':self.Ent_surface, 'ims':self.image_surface})
        self.manual_AS=False 


    def add_surface(self, curvature_radius:float, thickness:float, aperture_Radius:float, material:str, surface_type:str, k_conic:float=0, A_coeff:list=[], stop_iris= False, stop_size=None, surface_name:str=""): 
        if surface_type.lower() in ['sphere', 'spheric', 'spherical', 'sph']:
            surface_type='sph'
        
        elif surface_type.lower() in ['asphere', 'aspheric', 'aspherical', 'asph']:
            surface_type='asph'
        else:
            raise ValueError(f"Suface Type{surface_type} is not valid, it must be eihter sph or asph")
        

        surf_cnt=len(self.Surfaces.keys())
        
        for key in list(self.Surfaces.keys()):
            surf= self.Surfaces[key]
            if surf['surface_name']=='ims':
                surf_cnt -= 1
                self.Surfaces.pop(key)

        if surface_name=="":
            surface_name=f"surface{surf_cnt}"

        if stop_size==None:
            stop_size= aperture_Radius


        surface= {'surface_name':surface_name,
                  'radius': curvature_radius,
                  'thickness': thickness,
                  'aperture': aperture_Radius, 
                  'material': material, 
                  'type': surface_type,
                  'conic': k_conic,
                  'A_coefficient': A_coeff,
                  'stop':stop_iris,
                  'stop_size':stop_size,
                  'ast':False}
        

        
        self.Surfaces[f'surf{surf_cnt}']=surface
        self.Surfaces['ims']=self.image_surface


    def add_Lens_ZMX(self, lens:str, category:str=None, thickness_after: float= 0.0, reverse:bool= False): 
        Radiuses,Thickness,Materials, Aperature, Surface_Type, K_conic, Asph_Par= self.get_lens_parameters(lens, category)
        # print(K)
        if 'aspheric' not in Surface_Type:
            Thickness[-1]= thickness_after
            if reverse:
                Radiuses= (-array(Radiuses[::-1])).tolist()
                Aperature=Aperature[::-1]
                Surface_Type= Surface_Type[::-1]
        elif 'aspheric' in Surface_Type:
            idx= Surface_Type.index("aspheric")
            if Radiuses[idx]>0:
                Radiuses=Radiuses[idx:idx+2]
                Aperature=[Aperature[idx],Aperature[idx]]
                Thickness=Thickness[idx:idx+2]
                Thickness[-1]=thickness_after
                K_conic=K_conic[idx:idx+2]
                Asph_Par= Asph_Par[idx:idx+2]
                Surface_Type=Surface_Type[idx:idx+2]
                Materials=Materials[idx:idx+2]
            else:
                Radiuses=Radiuses[idx-1:idx+1]
                Aperature=[Aperature[idx],Aperature[idx]]
                Thickness=Thickness[idx-1:idx+1]
                Thickness[-1]=thickness_after
                K_conic=K_conic[idx-1:idx+1]
                Asph_Par= Asph_Par[idx-1:idx+1]
                Surface_Type=Surface_Type[idx-1:idx+1]
                Materials=Materials[idx-1:idx+1]

            if reverse:
                Asph_Par= [-array(i) for i in Asph_Par[::-1]]
                Radiuses= (-array(Radiuses[::-1])).tolist()
                Surface_Type= Surface_Type[::-1]
                K_conic= K_conic[::-1]
        
        for i in range(len(Surface_Type)):
            if Surface_Type[i]=='spheric':

                self.add_surface(Radiuses[i], Thickness[i], Aperature[i], Materials[i], Surface_Type[i])
            elif Surface_Type[i]=='aspheric':
                self.add_surface(Radiuses[i], Thickness[i], Aperature[i], Materials[i], Surface_Type[i], K_conic[i], Asph_Par[i])


        
    def set_Stop(self, aperture=10, surface_index=1):
        surf_keys= self.Surfaces.keys().tolist()
        
        if surface_index==0:
            self.Surfaces[surf_keys[1]].stop=True
            self.Surfaces[surf_keys[1]].stop_size= aperture
        elif surface_index>=len(self.Surfaces.keys())-1:
            self.Surfaces[surf_keys[-1]].ast=True
            self.Surfaces[surf_keys[-1]].stop_size= aperture
        else:
            self.Surfaces[surf_keys[surface_index]].stop=True
            self.Surfaces[surf_keys[surface_index]].stop_size= aperture



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

    def Lens_Layout(self, fig=None, ax=None,show_obj:bool=False, show_img:bool=True,AST_Active:bool=True, obj_col:str='orange' ,figsize=(10,4), xscale:str='symlog',yscale:str='symlog', Surfaces=None):
        
        if array([Surfaces==None]).all():
            Surfaces = self.Surfaces.copy(deep=True)
        
        surfaces_keys= Surfaces.keys().to_list()
        # if not any([Surfaces[sk].ast for sk in surfaces_keys ]):
        #     self.set_Ast(self.y0, surface_index=1)

        if fig==None or ax==None:
            fig,ax= subplots(figsize=figsize, tight_layout=True)

        S= Surfaces
        z_copy=S.loc[ 'thickness', 'obj']
        if not show_obj:
            S.loc[ 'thickness', 'obj']=0
        Z_plot= cumsum(S.loc['thickness', :].values)[:-1].tolist()+[0]
    
        idxs=argsort(Z_plot)
        Z_plot=array(Z_plot)[idxs]
        surf_prev=None
        state=False
        y_sag=[]
        z_sag=[]
        for i, (surf, z_plot) in enumerate(zip(Surfaces, Z_plot)):
            if surf in ['obj', 'ims']:
                if surf=='obj' and show_obj:
                    if self.h_obj!=0:
                        ax.vlines(z_plot,0,self.h_obj, color=obj_col, linewidth=5, alpha=.3)
                        ax.scatter(z_plot,self.h_obj, color=obj_col, linewidth=5)
                    else:
                        ax.scatter(z_plot,0, color=obj_col, linewidth=5)
                    y_sag=[]
                    z_sag=[]

        
                elif surf=='ims' and show_img:
                    
                    R= S[surf].radius
                    t= S[surf].thickness
                    ap= S[surf].aperture
                    mat= S[surf].material
                    typ= S[surf].type
                    k= S[surf].conic
                    As= S[surf].A_coefficient
                    ast= S[surf].ast
                    stop=S[surf].stop
                    stop_size=S[surf].stop_size
                    ax.vlines(z_plot+t,-10,10, color='k', linestyle='--')
                    y_sag=[]
                    z_sag=[]
                    
                    if stop and AST_Active:
                        if any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                            zz=.2
                            ax.vlines(z_plot,stop_size,stop_size+10,  linewidth=4,color='maroon', alpha=.3)
                            ax.vlines(z_plot,-stop_size,-stop_size-10,linewidth=4,color='maroon', alpha=.3)

                            ax.hlines(stop_size+zz,z_plot-0.5,t+z_plot+0.5, color='maroon', zorder=11)
                            ax.hlines(-stop_size-zz,z_plot-0.5,t+z_plot+0.5,color='maroon', zorder=11)
                            ax.vlines(z_plot,stop_size+zz,stop_size+10,      color='maroon', zorder=11)
                            ax.vlines(z_plot,-stop_size-zz,-stop_size-10,    color='maroon', zorder=11)
                        
                        else:
                            yyy= abs(R)-1e-10
                            if ap<abs(R):
                                yyy= ap
                            if typ=='asph':
                                z_plot_corr= self.sag_aspherical(yyy, R, k, As, z_plot)
                            else:
                                z_plot_corr= self.sag_spherical(yyy, R, z_plot)
                            # print(z_plot_corr, z_plot)
                            zz=.2
                            ax.vlines(z_plot_corr,stop_size,stop_size+10,  linewidth=4,color='maroon', alpha=.3)
                            ax.vlines(z_plot_corr,-stop_size,-stop_size-10,linewidth=4,color='maroon', alpha=.3)

                            ax.hlines(stop_size+zz,z_plot_corr-0.5,z_plot_corr+0.5, color='maroon', zorder=11)
                            ax.hlines(-stop_size-zz,z_plot_corr-0.5,z_plot_corr+0.5,color='maroon', zorder=11)
                            ax.vlines(z_plot_corr,stop_size+zz,stop_size+10,      color='maroon', zorder=11)
                            ax.vlines(z_plot_corr,-stop_size-zz,-stop_size-10,    color='maroon', zorder=11)
                    
            else:
                # print(dz)
                
                R= S[surf].radius
                t= S[surf].thickness
                ap= S[surf].aperture
                mat= S[surf].material
                typ= S[surf].type
                k= S[surf].conic
                As= S[surf].A_coefficient
                ast= S[surf].ast
                stop=S[surf].stop
                stop_size=S[surf].stop_size

                if typ=='sph' and R!=0:
                    y_sag= linspace(-ap, ap, 1001, endpoint=True)
                    z_sag= array([self.sag_spherical(ysag, R, z_plot) for ysag in y_sag])
                elif typ=='asph' and R!=0:
                    y_sag= linspace(-ap, ap, 1001, endpoint=True)
                    z_sag= array([self.sag_aspherical(ysag, R, k, As, z_plot) for ysag in y_sag])
                else:
                    ap_plot=ap
                    if ap>1e3:
                        ap_plot=10
                    
                    y_sag= linspace(-ap_plot, ap_plot, 1001)
                    z_sag= ones(1001)*z_plot
                if surf=='ent':
                    linestyle='--'
                    alpha=.3

                else:
                    linestyle='-'
                    alpha=1

                ax.plot(z_sag, y_sag, color='k', linestyle=linestyle,alpha=alpha)
                if stop and AST_Active:
                    if any([(R<=0 and mat.lower()=='air'),(R>=0 and mat.lower()!='air')]):
                        zz=.2
                        ax.vlines(z_plot,stop_size,stop_size+10,  linewidth=4,color='maroon', alpha=.3)
                        ax.vlines(z_plot,-stop_size,-stop_size-10,linewidth=4,color='maroon', alpha=.3)

                        ax.hlines(stop_size+zz,z_plot-0.5,z_plot+0.5, color='maroon', zorder=11)
                        ax.hlines(-stop_size-zz,z_plot-0.5,z_plot+0.5,color='maroon', zorder=11)
                        ax.vlines(z_plot,stop_size+zz,stop_size+10,      color='maroon', zorder=11)
                        ax.vlines(z_plot,-stop_size-zz,-stop_size-10,    color='maroon', zorder=11)
                        
                    else:
                        yyy= abs(R)-1e-10
                        if ap<abs(R):
                            yyy= ap
                        if typ=='asph':
                            z_plot_corr= self.sag_aspherical(yyy, R, k, As, z_plot)
                        else:
                            z_plot_corr= self.sag_spherical(yyy, R, z_plot)
                        # print(z_plot_corr, z_plot)
                        zz=.2
                        ax.vlines(z_plot_corr,stop_size,stop_size+10,  linewidth=4,color='maroon', alpha=.3)
                        ax.vlines(z_plot_corr,-stop_size,-stop_size-10,linewidth=4,color='maroon', alpha=.3)

                        ax.hlines(stop_size+zz,z_plot_corr-0.5,z_plot_corr+0.5, color='maroon', zorder=11)
                        ax.hlines(-stop_size-zz,z_plot_corr-0.5,z_plot_corr+0.5,color='maroon', zorder=11)
                        ax.vlines(z_plot_corr,stop_size+zz,stop_size+10,      color='maroon', zorder=11)
                        ax.vlines(z_plot_corr,-stop_size-zz,-stop_size-10,    color='maroon', zorder=11)



                if state:
                    ax.fill_betweenx(
                y_sag,         # vertical coordinate
                z_sag_prev,        # left boundary in z
                z_sag,        # right boundary in z
                color='lightblue', alpha=0.5)
                    state=False
                if mat.lower()!='air':
                    state=True
            
            surf_prev= surf
            y_sag_prev= y_sag
            z_sag_prev= z_sag 

        grid(True)
        S.loc[ 'thickness', 'obj']=z_copy
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        return fig, ax
    
