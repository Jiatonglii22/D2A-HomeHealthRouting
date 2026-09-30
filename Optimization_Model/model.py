import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

def build_model(nurse_avail, patient_appt, dist, w_c): 
    model = gp.Model('Patient Assignments')

    # parameters 
    nurses = nurse_avail['nurse_id'].unique() 
    days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']

    A = {} # nurse availability - 1 if nurse i is available on day d, 0 otherwise 
    start = {} # shift start time of nurse i on day d (None if not working) 
    end = {} # shift end time of nurse i on day d (None if not working) 

    for nurse in nurses: 
        A[nurse] = {} 
        start[nurse] = {} 
        end[nurse] = {} 

        for day in days: 
            match = nurse_avail[(nurse_avail['nurse_id'] == nurse) & (nurse_avail['day'] == day)]

            if len(match) > 0: 
                A[nurse][day] = int(match.iloc[0]['available'])
                start[nurse][day] = time_to_minutes(match.iloc[0]['start_time'])
                end[nurse][day] = time_to_minutes(match.iloc[0]['end_time'])
            else: 
                A[nurse][day] = 0 
                start[nurse][day] = 0 
                end[nurse][day] = 0  

    patients = patient_appt['patient_id'].unique() 

    P = {} # patient appointments - 1 if patient needs appointment on day d, 0 otherwise
    l = {} # service time for patient j 
    for patient in patients: 
        P[patient] = {} 
        l[patient] = {} 

        for day in days: 
            match = patient_appt[(patient_appt['patient_id'] == patient) & (patient_appt['day'] == day)]

            if len(match) > 0: 
                P[patient][day] = 1 
                l[patient][day] = float(match.iloc[0]['service_time']) # minutes 
            else: 
                P[patient][day] = 0 
                l[patient][day] = 0 

    cpg = 0.5 # cost per mileage 
    c = {} # distance from nurse i's home base to patient j × cpg 
    tau = {} # travel minutes from origin to destination 
    for _, row in dist.iterrows(): 
        origin = row['origin'] 
        dest = row['destination']

        miles = float(row['distance_miles']) 
        travel_minutes = int(row['travel_minutes'])

        if origin not in c: 
            c[origin] = {} 
            tau[origin] = {} 

        travel_cost = miles * cpg 

        c[origin][dest] = travel_cost 
        tau[origin][dest] = travel_minutes

    # decision variables 
    # x_ijd = 1 if nurse i is assigned to patient j on day d, 0 otherwise 
    x = model.addVars(
        nurses,
        patients,
        days, 
        vtype=GRB.BINARY,
        name='assign'
    )

    # z_ij = 1 if nurse i has ever been matched to patient j, 0 otherwise
    z = model.addVars(
        nurses, 
        patients, 
        vtype=GRB.BINARY, 
        name='continuity'
    )

    # constraints 
    # nurse availability 
    model.addConstrs(
        (x[i,j,d] <= A[i][d] 
         for i in nurses 
         for j in patients 
         for d in days), 
         name='nurse_availability'
    )

    # patient appointment fulfillment and cap at 1 assignment per patient 
    model.addConstrs(
        (gp.quicksum(x[i,j,d] for i in nurses) == P[j][d] 
         for j in patients 
         for d in days), 
         name='patient_appt_fulfullment'
    )

    # appointments stay within shift timings 
    model.addConstrs(
        (gp.quicksum(
            x[i,j,d] * (l[j][d] + tau[i][j]) 
            for j in patients
        ) <= end[i][d] - start[i][d] 
         for i in nurses 
         for d in days 
         if start[i][d] is not None and end[i][d] is not None), 
         name='hours_worked'
    )

    # limit shift hours to 8 hours (or 480 minutes) 
    model.addConstrs(
        (end[i][d] - start[i][d] <= 480 
         for i in nurses 
         for d in days
         if start[i][d] is not None and end[i][d] is not None), 
        name='limit_shift_hours'
    )

    # continuity of care 
    model.addConstrs(
        (x[i,j,d] <= z[i,j] 
         for i in nurses 
         for j in patients 
         for d in days), 
         name='continuity_of_care'
    )

    # objectives 
    travel_cost = gp.quicksum(
        c[i][j] * x[i,j,d]
        for i in nurses 
        for j in patients 
        for d in days
    )

    continuity_penalty = gp.quicksum(
        gp.quicksum(z[i,j] for i in nurses) - 1
        for j in patients 
    ) 

    model.setObjective(travel_cost + (w_c * continuity_penalty))

    return model, x, z 

def time_to_minutes(time_value): 
    if pd.isna(time_value): 
        return None 

    hours, minutes = map(int, str(time_value).split(':'))
    return (hours * 60) + minutes 