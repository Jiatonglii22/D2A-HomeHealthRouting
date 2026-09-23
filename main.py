from Optimization_Model.model import build_model 

model = build_model(); 

model.optimize() 

assert model.Status == 2, model.Status # ensures optimal solution is found 