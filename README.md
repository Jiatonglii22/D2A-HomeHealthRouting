# D2A-HomeHealthRouting
### Team Members: 
- Imama Zahoor 
- Jiatong Li 
- Esha Pandya 
- Vidhi Kothari 

*Decision question: For a home health agency serving Allegheny County, PA, how should weekly patient visits be assigned and sequenced across nurses, how many nurses of each skill type are required, and what is the operational cost of guaranteeing continuity of care?* 

## File Structure 
- Datasets: raw, forecasted, and processed datasets 
    - Processed datasets are ready to use as inputs 
- Optimization_Model: assignment and sequencing model 
    - Conducts sensitivity analysis and outputs results 
- Outputs: contains all outputs from optimization_model 
    - patient-nurse assignments per day 
    - daily and weekly schedules for each nurse 
    - route figures 
- Prediction_Model: uses existing data to perform demand allocation of patient appointments and locations 

## Setup: 
- Requires a Python version of 3.10 or higher 
- Download and unzip repository folder 
- Open a terminal in the main folder 
- Set up a virtual environment 
    - `python -m venv .venv` 
- Start virtual environment 
    - `source .venv/bin/activate` 
- Install required libraries 
    - `pip install -r requirements.txt` 

## Running the Models: 
- Run `main.py` 
    - `python main.py` 

## Outputs: 
- Navigate to `Outputs/` folder 
    - `nurse_daily_assignments.csv`: This csv contains the outputs of the assignment model. With columns as the day of the week and the row representing each nurse, each cell lists the IDs of the patients assigned to that nurse. It is the user readable output of the x_{ijd} binary decision variable. 
    - `route_schedule.csv`: This csv contains the output of the sequencing layer. Each row contains one stop in a nurse's stop, including the patient ID, patient appointment ID, day of the week, order in the stops, start and stop time of appointment, and drive time and mileage from the previous stop. 
    - `route_schedule_<day>.csv`: One csv per day is created in a similar format as the weekly schedule. 
    - `route_summary.csv`: This csv contains one row per nurse per day, with a list of patient IDs to easily extract the route. 
    - `unroutable_nurse_days.csv`: This csv will be populated if the sequencing model is unable to find a feasible route restricted to the nurse's shift hours. In an ideal situation, this file will be unpopulated. 
    - `staffing_sweep.csv`: This file contains the results of our sensitivty analysis, containing one row per combination of staff size and continuity of care weight. 

## Prediction Model

Agency-level demand allocation model is used to estimate agency-specific demand spatially using ACS need indicators, then use that estimated demand to create synthetic patients that Gurobi will route.

We're using UPMC Home Health Agency in Greater Pittsburgh Region as the agency we are allocating demand for in its serviced zip codes.

## Pipeline

The optimization runs in two layers:

1. **Assignment (Layer 1)**, `Optimization_Model/model.py`: a Gurobi MILP that decides which nurse sees which patient each day. It minimizes mileage cost from each nurse's home plus a continuity penalty `w_c` for every extra nurse a patient sees. Each nurse-day's visits, the drive to each one, and the drive home must fit within the shift.
2. **Sequencing (Layer 2)**, `Optimization_Model/sequencing.py`: takes those assignments and finds the visit order and timetable for every nurse-day.