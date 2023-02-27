from numba import jit
import numpy as np

class ice_cube_data:
    # everything for one batch only now
    def __init__(self,fbatch,width={'x':20,'y':20,'z':40},frac=0.1):
        import pandas as pd
        from pyarrow.parquet import ParquetFile
        import pyarrow as pa
        import numpy as np
        
        # Files for creating the data
        self.fsensor="/scratch/ICE-CUBE/sensor_geometry.csv" # default geometry
        self.fmeta=ParquetFile("/scratch/ICE-CUBE/train_meta.parquet") # meta data
        self.fbatch=fbatch # dictionary with all the batches
        
        self.sensorgeom_df = pd.read_csv(self.fsensor).set_index("sensor_id")# dataframe with sensor geometry
        
        
        for var in ['x','y','z']:
            self.sensorgeom_df[var] = ((self.sensorgeom_df[var] - self.sensorgeom_df[var].min()) / 
                                       (self.sensorgeom_df[var].max() - self.sensorgeom_df[var].min()))*width[var]
        
        self.sensorgeom_df=round(self.sensorgeom_df.astype("int")) # all x,y, as nearest integers
        
        
        self.xdim=(self.sensorgeom_df['x'].max()-self.sensorgeom_df['x'].min())+1
        self.ydim=(self.sensorgeom_df['y'].max()-self.sensorgeom_df['y'].min())+1
        self.zdim=(self.sensorgeom_df['z'].max()-self.sensorgeom_df['z'].min())+1
        
        self.sensorgeom=self.sensorgeom_df.to_dict()
        
        #charge depost and temporal data and meta data for only 1st batch (for now)
        self.batch= pd.read_parquet(self.fbatch['1']).query("(event_id%100==0) & (auxiliary==False)").drop(["auxiliary"],axis=1)
        #print(self.batch.loc[2800])
        #Just make a charge data, by summing charge deposits on the same sensor for each event, and create sensor_id again as column
        self.events=self.batch.drop(["time"],axis=1).groupby(["event_id","sensor_id"]).sum("charge").reset_index(level=[1])
        
        first_n_rows = next(self.fmeta.iter_batches(batch_size = 185806*100))#
        
        # Create meta data frame
        self.meta=pa.Table.from_batches([first_n_rows]).to_pandas()
        
        self.events['x']=self.events['sensor_id'].apply(self.getx)
        self.events['y']=self.events['sensor_id'].apply(self.gety)
        self.events['z']=self.events['sensor_id'].apply(self.getz)
        
        #for var in ['x','y','z']:
        #    self.events['x']=self.events[var]-self.events[var].min()
            
    def create_img_data(self):
        num_events=len(self.events.index.unique().to_list())
        self.img_data=np.zeros((num_events,self.xdim,self.ydim,self.zdim))
        self.labels=np.zeros((num_events,2))
        #print(f'creating image of shape:{self.img_data.shape}')
        
        @jit
        def runloop():
            for ievent,indexevent in enumerate(self.events.index.unique().to_list()):
                for x,y,z,c in self.events.loc[indexevent][["x","y","z","charge"]].values:
                    self.img_data[int(ievent),int(x),int(y),int(z)]=c
                    self.labels[int(ievent),0]=list(self.meta.query('event_id=='+str(indexevent))['azimuth'].values)[0]
                    self.labels[int(ievent),1]=list(self.meta.query('event_id=='+str(indexevent))['zenith'].values)[0]
                
        #print(f'created an image of shape:{self.img_data.shape} !!')
        return self.img_data,self.labels
        
    def getx(self,sensorid):
        return self.sensorgeom['x'][sensorid]
    def gety(self,sensorid):
        return self.sensorgeom['y'][sensorid]
    def getz(self,sensorid):
        return self.sensorgeom['z'][sensorid]

        
        