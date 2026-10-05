# Forecast dataset

One-week (Mon–Fri) input for the assignment model, built from the forecast demand for agency 397636 (UPMC Home Healthcare). Same layout as `Datasets/Dummy`, so `main.py` reads either one.

## Pipeline

```
python Prediction_Model/08_synthetic_patient_data.py   # forecast -> Processed/synthetic_*_397636.csv
python Prediction_Model/09_prepare_forecast_data.py               # -> Datasets/Forecast/*.csv
python DPrediction_Model/10_osrm_travel_matrix.py                  # -> Processed/travel_car.csv (needs internet)
python main.py                                         # -> Outputs/
```

`osrm_travel_matrix.py` reads `Datasets/Forecast` by default. To go back to the dummy week, run it with `--data-dir Datasets/Dummy` and set `DATA_DIR` in `main.py`.

## What changed from the forecast files

| Forecast column | Here |
|---|---|
| `day` = `Monday` … `Friday` | `Mon`, `Tues`, `Wed`, `Thurs`, `Fri` (must match `days` in model.py) |
| `service_time_minutes` | `service_time` |
| `latitude`, `longitude` | `lat`, `lon` |
| `zip` | `location` |
| `weekly_visits` | `visits_per_week` |

## Nurses are synthetic

The forecast has no staffing data, so `prepare_forecast_data.py` generates a seeded roster. The assumptions are constants at the top of the script:

- **Headcount:** the busiest day's workload (visit time plus an assumed 30 min of travel per visit) ÷ 480 min, × 1.3 slack. That gives 12 nurses for the current forecast.
- **Home base:** a ZIP drawn in proportion to forecast visits, placed within about 300 m of a forecast patient in that ZIP.
- **Shifts:** 8 h, starting between 7:00 and 9:00. About a quarter of nurses get one weekday off, redrawn until every day has at least 1.15 × the nurses it needs.
- **All nurses are RN.** The forecast has no visit types, so there are no skill restrictions.

`patient_appointments.csv` has no time windows, visit types or priorities. The forecast doesn't produce them, and the assignment model doesn't read them.
