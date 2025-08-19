
from numpy import array, tan, arctan, linspace, sqrt, inf, nan, arcsin, sin, pi, ndarray
from os.path import isdir
from os import listdir
from os.path import dirname,abspath
from warnings import catch_warnings, simplefilter
from matplotlib.pyplot import plot, subplots

class ZMXreader:
    def __init__(self):
        self.wl=0.58756
        self.zmx_dir= dirname(abspath(__file__)) + '//zmx_files'


    def get_lens_category(self):
            categories= [file for file in listdir(self.zmx_dir) if isdir(f'{self.zmx_dir}//{file}')]
            return categories
        
    def get_lens_list(self, category:str):
        category= category.lower()
        if category in self.get_lens_category():
            return [file.split('.')[0].upper() for file in listdir(f'{self.zmx_dir}//{category}') if file.lower().endswith('zmx')]
        
        else: raise ValueError(f'Lens Category:{category} not availble, you can choose form {self.get_lens_category()}')

    def All_Lenses_Printing(self):
        for cat in self.get_lens_category():
            print(cat,'===>', self.get_lens_list(cat))
            

    def lens_finder(self, lens:str, Print:bool= False):
        lens= lens.upper()
        categories= self.get_lens_category()
        state= False
        categories_res=[]
        for category in categories:
            if lens in self.get_lens_list(category):
                if Print:
                    print(f'Matching found, Category: {category}, Lens: {lens}')
                categories_res.append(category)
                state= True
                return categories_res
            
        if not state:
            raise ValueError (f'No Matching found!')



    def zmxLens_read(self, lens:str, category:str=None):
        if category==None:
            category= self.lens_finder(lens)[0]

        if lens.upper() not in self.get_lens_list(category):
            raise ValueError(f'Lens not Available, choose from {self.get_lens_list()}')
        
        with open(r'{:}//{:}.ZMX'.format(f'{self.zmx_dir}//{category}', lens), 'r', encoding= 'Latin1') as file:
            for line in file:
                if line.split()[0]!='VERS':
                    encoding= 'utf-16-le'
                    break
                else:
                    encoding= 'Latin1'
                    break
                
        with open(r'{:}//{:}.ZMX'.format(f'{self.zmx_dir}//{category}', lens), 'r', encoding= encoding) as file:
            Results=[]
            surface= {}
            st= False
            for line in file:
                if 'SURF' in line and st:
                    st= False
                    if 'Glass' not in surface:
                        surface['Glass']= 'AIR'
                    Results.append(surface)

                if 'SURF' in line and not st:
                    st=True
                    surface= {}
                    surface['Surface']= int(line.split()[1])
                
                elif 'COMM' in line:
                    surface['Comment']= line.split()[1]
                
                elif 'TYPE' in line:
                    if line.split()[1]=='STANDARD':
                        surface['Surf_Type']= 'spheric'
                    elif line.split()[1]=='EVENASPH':
                        Par=[]
                        surface['Surf_Type']= 'aspheric'
                
                elif 'CURV' in line:
                    if float(line.split()[1])==0:
                        surface['Radius']= 0
                    else:
                        surface['Radius']= round(1/float(line.split()[1]),6)
                
                elif 'DISZ' in line:
                    if line.split()[1]== 'INFINITY':
                        surface['thickness']= 1e20
                    else:
                        surface['thickness']= float(line.split()[1]) 
                
                elif 'GLAS' in line:
                    surface['Glass']= line.split()[1]
                
                elif 'DIAM' in line:
                    surface['ARadius']= float(line.split()[1])
                
                elif 'PARM' in line:
                    if float(line.split()[1]) != 1 and float(line.split()[1])<=6:
                        Par.append(float(line.split()[2]))
                        surface['Asph_Par']= Par

                elif 'CONI' in line:
                    surface['k']= float(line.split()[1])

            if 'Glass' not in surface:
                        surface['Glass']= 'AIR'
            Results.append(surface)
            return Results
        
    def get_lens_parameters(self, lens:str,category: str = None):
        res= self.zmxLens_read(lens,category)[1:-1]
        R= [r['Radius'] for r in res]
        T= [r['thickness'] for r in res]
        Surface_Type= [r['Surf_Type'] for r in res]
        glass= [r['Glass'] for r in res]
        Aperature= [r['ARadius'] for r in res]
        K=[]
        Asph_Par=[]
        for i in range (len(res)):
            if res[i]['Surf_Type']=='aspheric':
                K.append(res[i]['k'])
                Asph_Par.append(res[i]['Asph_Par'])
            else:
                K.append(0)
                Asph_Par.append([])

        if 'aspheric' in Surface_Type:
            for i in range (len(Surface_Type)):
                if Surface_Type[i]== 'aspheric':
                    if R[i]>0:
                        Aperature[i+1]= Aperature[i]
                    else:
                        Aperature[i-1]= Aperature[i]

            return R,T,glass, Aperature, Surface_Type, K, Asph_Par
        return R,T,glass, Aperature, Surface_Type, 0,[]

