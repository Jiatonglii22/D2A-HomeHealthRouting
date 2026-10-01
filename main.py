import pandas as pd 
from pathlib import Path 
from Optimization_Model.model import build_model 

patients_df = pd.read_csv('./Datasets/Dummy/patient_appointments.csv')
nurses_df = pd.read_csv('./Datasets/Dummy/nurse_availability.csv')
distance_df = pd.read_csv('./Datasets/Processed/travel_car.csv')

w_c = 0.5

model, x, z = build_model(nurses_df, patients_df, distance_df, w_c); 

model.optimize() 

assert model.Status == 2, model.Status # ensures optimal solution is found 

# output directory 
output_dir = Path("./Outputs")
output_dir.mkdir(parents=True, exist_ok=True)

days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'] 

# extract assignments from model 
assignments = [(i, j, d) for (i, j, d), var in x.items() if var.X > 0]

# nurse assignments by day (output to CSV) 
nurse_schedule = {
    nurse: {day: [] for day in days} 
    for nurse in nurses_df['nurse_id'].unique() 
}

for nurse, patient, day in assignments: 
    nurse_schedule[nurse][day].append(patient)

schedule_rows = [] 
for nurse, schedule in nurse_schedule.items(): 
    row = {'Nurse_ID': nurse}

    for day in days: 
        row[day] = ', '.join(sorted(schedule[day]))

    schedule_rows.append(row)

schedule_df = pd.DataFrame(schedule_rows, columns=['Nurse_ID'] + days)
schedule_df.to_csv(output_dir/'nurse_daily_assignments.csv', index=False) 

# nurse-patient continuity penalty (output to CSV) 
patient_nurses = {
    patient: set()
    for patient in patients_df['patient_id'].unique() 
}

for nurse, patient, day in assignments: 
    patient_nurses[patient].add(nurse)

continuity_rows = [] 
for patient, nurses in patient_nurses.items(): 
    num_nurses = len(nurses)

    if num_nurses > 1: 
        continuity_rows.append({
            'Patient_ID': patient, 
            'Nurse_IDs': ', '.join(sorted(nurses)), 
            'Continuity Penalty': max(0, num_nurses-1), 
            'Weighted_Penalty': w_c * max(0, num_nurses-1)
        })

continuity_df = pd.DataFrame(continuity_rows) 
continuity_df.to_csv(output_dir/'patient_continuity.csv', index=False)