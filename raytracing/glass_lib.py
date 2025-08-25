from numpy import sqrt, shape, array
from os import listdir
from os.path import dirname,abspath
from scipy.interpolate import interp1d


class Glass:
    def __init__(self):
        self.encoding= 'latin1'
        self.encoding_hoya= 'utf-16-le'
        self.agf_dir= dirname(abspath(__file__)) + '//glass_agf'

    def get_glass_provider(self):
        providers= [str(prvd.split('.agf')[0]) for prvd in listdir(r"{:}".format(self.agf_dir)) if prvd.endswith('.agf')]
        return providers
    
    def get_glass_list(self, provider:str):
        provider= provider.lower()
        
        if provider in self.get_glass_provider():
            Glass_Name=[]
            encoding= self.encoding
            if provider=='hoya':
                encoding= self.encoding_hoya

            with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                for line in file:
                    if line.startswith('NM'):
                        res= line.split(' ')
                        Glass_Name.append(str(res[1]))

            return Glass_Name
        else: raise ValueError(f'glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')

    def glass_finder(self, glass:str, Print:bool=False):
        glass= glass.upper()
        providers= self.get_glass_provider()
        state= False
        provider_res=[]
        for provider in providers:
            if glass in self.get_glass_list(provider):
                if Print:
                    print(f'Matching found, Provider: {provider}, glass: {glass}')
                provider_res.append(provider)
                state= True
                return provider_res
            
        if not state:
            raise ValueError(f'No Matching found!')
        

    def get_Nd(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()

        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya

                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            res= line.split(' ')
                            return float(res[4])
                        
            else: raise ValueError(f'glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')
    
    def get_abbe_number_datasheet(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
        
        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya

                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            res= line.split(' ')
                            return float(res[5])
                        
            else: raise ValueError(f'Glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')



    def get_abbe_number_cal(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
        
        wl_f= 0.4861327#µm
        wl_c= 0.6562725 #µm
        wl_d= 0.5875618 #µm
        wl_min, wl_max= self. get_wavelength_range(glass, provider)
        
        if (wl_f > wl_max or  wl_f < wl_min ) or (wl_c > wl_max or  wl_c < wl_min ):
            return 0
        
        Nd= self.get_Nd(glass, provider)
        Nf= self.get_Ref_Index(wl_f, glass, provider)
        Nc= self.get_Ref_Index(wl_c, glass, provider)
        
        return (Nd-1)/(Nf-Nc)


    def get_general_glass_comment(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
        
        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya

                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    st=False
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            st=True
                        elif line.startswith('GC') and st:
                            print(line)                   
                            break
            else: raise ValueError(f'Glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')
    

    def get_internal_transmission(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
        
        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya
                
                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    Transmission=[]
                    thickness=[]
                    Wave_length=[]
                    st=False
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            st=True
                        elif line.startswith('IT') and st:

                            res= line.split()[1:]
                            Wave_length.append(float(res[0]))
                            Transmission.append(float(res[1]))
                            thickness.append(float(res[2]))

                        elif line.startswith('NM') and st:
                            return Transmission, Wave_length, thickness
            
            else: raise ValueError(f'Glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')
    
    def get_wavelength_range(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
        
        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya

                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    st=False
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            st=True
                        elif line.startswith('LD') and st:
                            res= line.split()[1:]
                            wl_min= float(res[0])
                            wl_max= float(res[1])                      
                            return wl_min, wl_max
            
            else: raise ValueError(f'Glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')

    def get_disp_formula_num(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
                   
        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya

                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            res= line.split(' ')
                            return int(res[2])
                        
            else: raise ValueError(f'Glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')
    
    def get_dispersion_coefficient(self, glass:str, provider:str):
        provider= provider.lower()
        glass= glass.upper()
        
        if provider in self.get_glass_provider():
            if glass in self.get_glass_list(provider):
                encoding= self.encoding
                if provider=='hoya':
                    encoding= self.encoding_hoya
                
                with open(r'{:}//{:}.agf'.format(self.agf_dir, provider), 'r', encoding= encoding) as file:
                    st=False
                    for line in file:
                        if line.startswith('NM') and glass == line.split(' ')[1]:
                            st=True
                        elif line.startswith('CD') and st:
                            res= line.split()[1:]
                            cd= [float(a) for a in res]
                            return list(filter(lambda num: num != 0, cd))
                        
            else: raise ValueError(f'Glass: {glass} not availble for this provider ({provider}), you can choose form {self.get_glass_list(provider)}')
        else: raise ValueError(f'Glass Privder:{provider} not availble, you can choose form {self.get_glass_provider()}')



    def get_disp_formula_type(self, formula_number:int):
            if formula_number>13 or formula_number<1:
                raise ValueError('choose from (1-13); 1: Schott, 2: Sellmeier1, 3: Herzberger, 4: Sellmeier2, 5: Conrady, 6: Sellmeier3, 7: Handbook1_formula, 8: Handbook2_formula 9: Sellmeier4, 10: Extended1_formula, 11: Sellmeier5, 12:Extended2_formula, 13:Extended3_formula')

            if formula_number==1:
                return'Schott'
            if formula_number==2:
                return'Sellmeier1'
            if formula_number==3:
                return'Herzberger'
            if formula_number==4:
                return'Sellmeier2'
            if formula_number==5:
                return'Conrady'
            if formula_number==6:
                return'Sellmeier3'
            if formula_number==7:
                return'Handbook1_formula'
            if formula_number==8:
                return'Handbook2_formula'
            if formula_number==9:
                return'Sellmeier4'
            if formula_number==10:
                return'Extended1_formula'
            if formula_number==11:
                return'Sellmeier5'
            if formula_number==12:
                return'Extended2_formula'
            if formula_number==13:
                return'Extended3_formula'
            
    def get_Ref_Index(self,wl_um:float, glass:str, provider:str | None):
        n= []
        if not shape(wl_um):
            WL= [wl_um]
        else:
            WL= wl_um
        
        # print(glass,glass.lower(),glass.lower()!='air')
        if glass.lower()!='air':
            if provider==None:
                provider= self.glass_finder(glass)[0].lower()
            else:
                provider= provider.lower()
            glass= glass.upper()
            
            formula_number= self.get_disp_formula_num(glass, provider)
            C= self.get_dispersion_coefficient(glass, provider)
            wl_min, wl_max= self. get_wavelength_range(glass, provider)

            for wl in WL:
                
                if wl> wl_max or wl<wl_min and formula_number!=14:
                    raise ValueError(f'Out of the Wavelength Range: you can have between {wl_min}µm and {wl_max}µm ')
                elif formula_number==1:
                    n.append(sqrt(self.Schott(wl, C)))
                
                elif formula_number==2:
                    n.append(sqrt(self.Sellmeier1(wl, C[::2], C[1::2])))
                
                elif formula_number==3:
                    n.append(sqrt(self.Herzberger(wl, *C)))
                
                elif formula_number==4:
                    raise NotImplemented('the function not implemented yet')

                elif formula_number==5:
                    raise NotImplemented('the function not implemented yet')

                elif formula_number==6:
                    n.append(sqrt(self.Sellmeier3(wl, C[::2], C[1::2])))

                elif formula_number==7:
                    n.append(sqrt(self.Handbook1_formula(wl, *C)))

                elif formula_number==8:
                    n.append(sqrt(self.Handbook2_formula(wl, *C)))
                
                elif formula_number==9:
                    raise NotImplemented('the function not implemented yet')

                elif formula_number==10:
                    raise NotImplemented('the function not implemented yet')

                elif formula_number==11:
                    raise NotImplemented('the function not implemented yet')

                elif formula_number==12:
                    n.append(sqrt(self.Extended2_formula(wl, C)))

                elif formula_number==13:
                    n.append(sqrt(self.Extended3_formula(wl, C)))
                    

            if len(n)==1:
                n=n[0]
            return n
        
        else:
            for wl in WL:
                n.append(sqrt(self.Air_formula(wl)))
            if len(n)==1:
                n=n[0]
            return n
    
    def Schott(self, wl_um:float, A:list):
        wl= wl_um
        n2= A[0]+ A[1]*wl**2
        for i, a in enumerate(A[2:]):
            n2+=a*wl**(-(i+1)*2)
        return n2

    def Sellmeier1(self, wl_um:float, K:list, L:list):
        wl=wl_um
        n2= 1
        for (k, l) in zip(K, L):
            n2+= k*wl**2/(wl**2-l)
        return n2

    def Sellmeier2(self, wl_um:float, wl1_um:float,wl2_um:float, A:float, B:list):
        wl=wl_um
        wl1= wl1_um
        wl2= wl2_um
        return 1+A+ (B[0]*wl**2/(wl**2-wl1**2))+ (B[1]*wl**2/(wl**2-wl2**2))

    def Sellmeier3(self, wl_um:float, K:list, L:list):
        wl=wl_um
        n2= 1
        for (k, l) in zip(K, L):
            n2+= k*wl**2/(wl**2-l)
        return n2

    def Sellmeier4(self, wl_um:float, A:float, B:float, C:float, D:float, E: float):
        wl=wl_um
        return A+ (B*wl**2/(wl**2-C))+ (D*wl**2/(wl**2-E))

    def Sellmeier5(self, wl_um:float, K:list, L:list):
        wl=wl_um
        n2= 1
        for (k, l) in zip(K, L):
            n2+= k*wl**2/(wl**2-l)
        return n2

    def Herzberger(self, wl_um:float, A:float, B:float, C:float, D:float, E: float, F:float):
        wl= wl_um
        L= 1/(wl**2-0.028)
        return (A+(B*L)+(C*L**2)+(D*wl**2)+(E*wl**4)+(F*wl**6))**2

    def Conrady(self, wl_um:float, n0:float, A:float, B:float):
        wl= wl_um
        return (n0+A/wl+B/wl**3.5)**2


    def Handbook1_formula(self, wl_um:float, A:float, B:float, C:float, D:float):
        wl= wl_um
        return A+B/(wl**2-C)-D*wl**2

    def Handbook2_formula(self, wl_um:float, A:float, B:float, C:float, D:float):
        wl= wl_um
        return A+B*wl**2/(wl**2-C)-D*wl**2

    def Extended1_formula(self, wl_um:float, a:list):
        wl= wl_um
        n2= a[0]+a[1]*wl**2
        for i, a in enumerate(a[2:]):
            n2+= a*wl**(-(i+1)*2)
        return n2

    def Extended2_formula(self, wl_um:float, a:list):
        wl= wl_um
        while True:
            if len(a)!=8:
                a.append(0)
            elif len(a)>8:
                break
            else:
                break

        n2= a[0]+a[1]*wl**2+a[6]*wl**4+a[7]*wl**6
        for i, a in enumerate(a[2:-2]):
            n2+= a*wl**(-(i+1)*2)
        return n2

    def Extended3_formula(self, wl_um:float, a:list):
        wl= wl_um
        n2= a[0]+a[1]*wl**2+a[2]*wl**4
        for i, a in enumerate(a[3:]):
            n2+= a*wl**(-(i+1)*2)
        return n2
    
    def Air_formula(self, wl_um:float):
        if wl_um>=0.185 and wl_um<=1.7:
            return (1+8.06051e-5+(2.480990e-2/(132.274-wl_um**-2))+(1.74557e-4/(39.32957-wl_um**-2)))**2
    
        elif wl_um>=7.5 and wl_um<=14.1:
            wl= array([ 7.5 ,  7.55,  7.6 ,  7.65,  7.7 ,  7.75,  7.8 ,  7.85,  7.9 ,
                            7.95,  8.  ,  8.05,  8.1 ,  8.15,  8.2 ,  8.25,  8.3 ,  8.35,
                            8.4 ,  8.45,  8.5 ,  8.55,  8.6 ,  8.65,  8.7 ,  8.75,  8.8 ,
                            8.85,  8.9 ,  8.95,  9.  ,  9.05,  9.1 ,  9.15,  9.2 ,  9.25,
                            9.3 ,  9.35,  9.4 ,  9.45,  9.5 ,  9.55,  9.6 ,  9.65,  9.7 ,
                            9.75,  9.8 ,  9.85,  9.9 ,  9.95, 10.  , 10.05, 10.1 , 10.15,
                            10.2 , 10.25, 10.3 , 10.35, 10.4 , 10.45, 10.5 , 10.55, 10.6 ,
                            10.65, 10.7 , 10.75, 10.8 , 10.85, 10.9 , 10.95, 11.  , 11.05,
                            11.1 , 11.15, 11.2 , 11.25, 11.3 , 11.35, 11.4 , 11.45, 11.5 ,
                            11.55, 11.6 , 11.65, 11.7 , 11.75, 11.8 , 11.85, 11.9 , 11.95,
                            12.  , 12.05, 12.1 , 12.15, 12.2 , 12.25, 12.3 , 12.35, 12.4 ,
                            12.45, 12.5 , 12.55, 12.6 , 12.65, 12.7 , 12.75, 12.8 , 12.85,
                            12.9 , 12.95, 13.  , 13.05, 13.1 , 13.15, 13.2 , 13.25, 13.3 ,
                            13.35, 13.4 , 13.45, 13.5 , 13.55, 13.6 , 13.65, 13.7 , 13.75,
                            13.8 , 13.85, 13.9 , 13.95, 14.  , 14.05, 14.1 ])

            n= array([1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 , 1.0002727 ,
                        1.0002727 , 1.0002727 , 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027269, 1.00027269, 1.00027269, 1.00027269,
                        1.00027269, 1.00027268, 1.00027268, 1.00027268, 1.00027268,
                        1.00027268, 1.00027268, 1.00027268, 1.00027268, 1.00027268,
                        1.00027268, 1.00027268, 1.00027267, 1.00027267, 1.00027267,
                        1.00027267, 1.00027267, 1.00027267, 1.00027267, 1.00027267,
                        1.00027266, 1.00027266, 1.00027266, 1.00027266, 1.00027266,
                        1.00027265, 1.00027265, 1.00027265, 1.00027265, 1.00027265,
                        1.00027264, 1.00027264, 1.00027264])

            func= interp1d(wl, n)
            return func(wl_um)**2
        
        else: 
            return 1.00027




    # def get_group_velocity_disp(self):
    #     raise NotImplemented('the function not implemented yet')
    
    # def get_chromatic_dispersion(self):
    #     raise NotImplemented('the function not implemented yet')
    
    # def get_brewster_angle(self):
    #     raise NotImplemented('the function not implemented yet')
    
    # def get_reflectance(self):
    #     raise NotImplemented('the function not implemented yet')
    

 