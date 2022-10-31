import numpy as np
import json
import pickle
import config



class AutoModel():
    def __init__(self,symboling, normalized_losses, wheel_base, length,
       width, height, curb_weight, engine_size, bore, stroke,
       compression_ratio, horsepower, peak_rpm, city_mpg,
       highway_mpg, fuel_type,aspiration, num_of_doors, body_style,
       drive_wheels,engine_location, engine_type,
       num_of_cylinders, fuel_system):

       self.symboling=symboling,        
       self.normalized_losses=normalized_losses,
       self.wheel_base=wheel_base,      
       self.length=length,
       self.width=width,                
       self.height=height,
       self.urb_weight=curb_weight,    
       self.engine_size=engine_size,
       self.bore=bore,                  
       self.stroke=stroke,
       self.compression_ratio=compression_ratio,
       self.horsepower=horsepower,
       self.peak_rpm=peak_rpm,
       self.city_mpg=city_mpg,
       self.highway_mpg=highway_mpg,
       self.fuel_type='fuel-type_'+fuel_type,
       self.aspiration='aspiration_'+aspiration,
       self.num_of_doors='num-of-doors_'+num_of_doors,
       self.body_style='body-style_'+body_style,
       self.drive_wheels='drive-wheels_'+drive_wheels,
       self.engine_location='engine-location_'+engine_location,
       self.engine_type='engine-type_'+engine_type,
       self.num_of_cylinders='num-of-cylinders_'+num_of_cylinders,
       self.fuel_system='fuel-system_'+fuel_system
    
    def load_files(self):
        with open (config.JSON_FILE_PATH,'r') as f:
            self.json_file=json.load(f)
        with open (config.MODEL_FILE_PATH,'rb') as f:
            self.model=pickle.load(f)

    
    def get_pred_price(self):
        self.load_files()

        fuel_type_index=self.json_file['columns'].index(self.fuel_type)
        aspiration_index=self.json_file['columns'].index(self.aspiration)
        num_of_doors_index=self.json_file['columns'].index(self.num_of_doors)
        body_style_index=self.json_file['columns'].index(self.body_style)
        drive_wheels_index=self.json_file['columns'].index(self.drive_wheels)
        engine_location_index=self.json_file['columns'].index(self.engine_location)
        engine_type_index=self.json_file['columns'].index(self.engine_type)
        num_of_cylinders_index=self.json_file['columns'].index(self.num_of_cylinders)
        fuel_system_index=self.json_file['columns'].index(self.fuel_system)

        test_array=np.zeros(len(self.json_file['columns']))

        test_array[0]=  self.symboling
        test_array[1]=  self.normalized_losses
        test_array[2]=  self.wheel_base     
        test_array[3]=  self.length
        test_array[4]=  self.width                
        test_array[5]=  self.height
        test_array[6]=  self.urb_weight  
        test_array[7]=  self.engine_size
        test_array[8]=  self.bore                 
        test_array[9]=  self.stroke
        test_array[10]= self.compression_ratio
        test_array[11]= self.horsepower
        test_array[12]= self.peak_rpm
        test_array[13]= self.city_mpg
        test_array[14]= self.highway_mpg
        test_array[fuel_type_index]=  1
        test_array[aspiration_index]= 1
        test_array[num_of_doors_index]=1
        test_array[body_style_index]=1
        test_array[drive_wheels_index]=1
        test_array[engine_location_index]=1
        test_array[engine_type_index]=1
        test_array[num_of_cylinders_index]=1
        test_array[fuel_system_index]=1

        pred_car_price=self.model.predict([test_array])[0]
        return pred_car_price












