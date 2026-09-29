import pandas as pd 
from Optimization_Model.model import build_model 

patients_df = pd.read_csv('./Datasets/Dummy/patient_appointments.csv')
nurses_df = pd.read_csv('./Datasets/Dummy/nurse_availability.csv')
distance_df = pd.read_csv('./Datasets/Dummy/travel_matrix.csv')

model, x, z = build_model(nurses_df, patients_df, distance_df, 0.5); 

model.optimize() 

assert model.Status == 2, model.Status # ensures optimal solution is found 
print(model.Status == 2) 
for v in model.getVars(): 
    print(f"{v.VarName} = {v.X}")