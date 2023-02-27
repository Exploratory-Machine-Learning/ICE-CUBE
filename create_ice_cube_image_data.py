from numba import jit
import numpy as np

class ice_cube_data:
    # everything for one batch only now
    def __init__(self,fbatch,width={'x':20,'y':20,'z':40}):
        import pandas as pd
        from pyarrow.parquet import ParquetFile
        import pyarrow as pa
        import numpy as np
        
        # Files for creating the data
        self.fsensor="/scratch/ICE-CUBE/sensor_geometry.csv" # default geometry
        self.fmeta=ParquetFile("/scratch/ICE-CUBE/train_meta.parquet") # meta data
        self.fbatch=fbatch # dictionary with all the batches
        
        self.sensorgeom_df_orig = pd.read_csv(self.fsensor).set_index("sensor_id")# dataframe with sensor geometry
        
        self.sensorgeom_df=self.sensorgeom_df_orig.copy()
        for var in ['x','y','z']:
            self.sensorgeom_df[var] = ((self.sensorgeom_df[var] - self.sensorgeom_df[var].min()) / 
                                       (self.sensorgeom_df[var].max() - self.sensorgeom_df[var].min()))*width[var]
        
        self.sensorgeom_df=round(self.sensorgeom_df.astype("int")) # all x,y, as nearest integers
        
        
        self.xdim=(self.sensorgeom_df['x'].max()-self.sensorgeom_df['x'].min())+1
        self.ydim=(self.sensorgeom_df['y'].max()-self.sensorgeom_df['y'].min())+1
        self.zdim=(self.sensorgeom_df['z'].max()-self.sensorgeom_df['z'].min())+1
        
        self.sensorgeom=self.sensorgeom_df.to_dict()
        self.sensorgeomorig=self.sensorgeom_df_orig.to_dict()
        
        #charge depost and temporal data and meta data for only 1st batch (for now)
        self.batch= pd.read_parquet(self.fbatch['1']).query("(event_id%100==0) & (auxiliary==False)").drop(["auxiliary"],axis=1)
        #print(self.batch.loc[2800])
        #Just make a charge data, by summing charge deposits on the same sensor for each event, and create sensor_id again as column
        self.realevents=self.batch.drop(["time"],axis=1).groupby(["event_id","sensor_id"]).sum("charge").reset_index(level=[1])
        
        first_n_rows = next(self.fmeta.iter_batches(batch_size = 185806*100))#
        
        # Create meta data frame
        self.meta=pa.Table.from_batches([first_n_rows]).to_pandas()
        
        self.events=self.realevents.copy()
        self.events['x']=self.events['sensor_id'].apply(self.getx)
        self.events['y']=self.events['sensor_id'].apply(self.gety)
        self.events['z']=self.events['sensor_id'].apply(self.getz)
        
        self.realevents['x']=self.realevents['sensor_id'].apply(self.getxo)
        self.realevents['y']=self.realevents['sensor_id'].apply(self.getyo)
        self.realevents['z']=self.realevents['sensor_id'].apply(self.getzo)
        
        #for var in ['x','y','z']:
        #    self.events['x']=self.events[var]-self.events[var].min()
            
    def create_img_data(self):
        num_events=len(self.events.index.unique().to_list())
        self.img_data=np.zeros((num_events,self.xdim,self.ydim,self.zdim))
        self.labels=np.zeros((num_events,2))
        self.eventdict={}
        for ievent,indexevent in enumerate(self.events.index.unique().to_list()):
            self.eventdict[indexevent]=ievent

        myevents=self.events.copy()
        myevents = myevents.reset_index()
        
        def fill_image_data(row):
                self.img_data[self.eventdict[int(row.event_id)], int(row.x), int(row.y), int(row.z)] = row.charge
                self.labels[self.eventdict[row.event_id], 0] = self.meta.loc[ievent, "azimuth"]
                self.labels[self.eventdict[row.event_id], 1] = self.meta.loc[ievent, "zenith"]
                
        myevents.apply(fill_image_data, axis=1)
        
                
        #print(f'created an image of shape:{self.img_data.shape} !!')
        return self.img_data,self.labels
        
    def getx(self,sensorid):
        return self.sensorgeom['x'][sensorid]
    def gety(self,sensorid):
        return self.sensorgeom['y'][sensorid]
    def getz(self,sensorid):
        return self.sensorgeom['z'][sensorid]
    
    def getxo(self,sensorid):
        return self.sensorgeomorig['x'][sensorid]
    def getyo(self,sensorid):
        return self.sensorgeomorig['y'][sensorid]
    def getzo(self,sensorid):
        return self.sensorgeomorig['z'][sensorid]

        
        