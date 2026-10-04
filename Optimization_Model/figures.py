"""
Presentation figures, built from the CSVs the pipeline saves (no re-solving).

  staffing_sweep         weekly route cost vs % of patients who keep one nurse, one line per roster size
  cost_by_roster         weekly route cost per roster size: lowest-cost schedule vs full continuity
  nurse_time_breakdown   how each nurse's week splits into visits, driving and spare shift time
  route_map_<day>        each nurse's route that day, one small map per nurse
  route_map_all_<day>    every nurse's route that day on one map, one colour per nurse, over a street basemap

Each figure is saved to Outputs/figures/ as a 300 dpi PNG (slides, print) and an SVG (scales to any size).

Inputs
  Outputs/staffing_sweep.csv                       from Optimization_Model/sweep.py
  Outputs/route_summary.csv, route_schedule.csv    from Optimization_Model/sequencing.py
  Datasets/Processed/synthetic_patients_397636.csv patient locations
  Datasets/Forecast/nurses.csv                     nurse home locations (the synthetic roster; not in Processed)
  Datasets/Raw/AlleghenyCounty__ZipCodeBoundaries202607.shp   ZIP outlines for route_map_all (needs geopandas)

route_map_all draws street tiles with contextily (downloads tiles, needs internet). Without contextily or a
connection it falls back to the ZIP outlines alone.

Run from the repo root, after main.py, sequencing and the sweep:  python -m Optimization_Model.figures
"""

import math
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from Optimization_Model import sequencing as seq

OUTPUT_DIR = Path(seq.OUTPUT_DIR)
FIGURES_DIR = OUTPUT_DIR / 'figures'
SWEEP = OUTPUT_DIR / 'staffing_sweep.csv'
ROUTE_SUMMARY = OUTPUT_DIR / 'route_summary.csv'
ROUTE_SCHEDULE = OUTPUT_DIR / 'route_schedule.csv'
PATIENTS = Path('./Datasets/Processed/synthetic_patients_397636.csv')
NURSES = Path('./Datasets/Forecast/nurses.csv')
ZIP_BOUNDARIES = Path('./Datasets/Raw/AlleghenyCounty__ZipCodeBoundaries202607.shp')

LOWEST_COST_W_C = 0 # sweep weight for "ignore continuity"
FULL_CONTINUITY_W_C = 500 # sweep weight that forces one nurse per patient

# palette (validated for colour-vision deficiency and contrast on a white surface)
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e6e5e1'
AXIS = '#c3c2b7'
BLUE = '#2a78d6' # series 1
ORANGE = '#eb6834' # series 2
SPARE = '#d9d8d3' # remainder / context, deliberately recessive
ROSTER_COLORS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'] # one blue, light -> dark with roster size
# categorical hues in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red); nurses past the
# 8th reuse them with a dashed line, so no two nurses share colour AND line style
NURSE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', color=INK, fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2)


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300) # sharp on projectors and in print
    fig.savefig(FIGURES_DIR / f'{name}.svg') # vector: scales to any size in PowerPoint / Google Slides
    plt.close(fig)
    print(f'  {FIGURES_DIR / name}.png / .svg')


def usable_runs(sweep):
    """Runs from rosters that give a drivable schedule at EVERY w_c (a solution and no unroutable nurse-days).

    Keeping whole rosters, not single runs, means each line averages the same rosters at every point,
    so a point never moves just because a roster dropped out there.
    """
    solved = sweep['status'].str.startswith(('optimal', 'time limit'))
    good = (solved & (sweep['unroutable_days'] == 0)).groupby([sweep['nurse_count'], sweep['seed']]).transform('all')
    return sweep[good]


# ---------------------------------------------------------------- sweep figures

def plot_staffing_sweep(sweep):
    """Weekly route cost vs % of patients who keep one nurse, one line per roster size (averaged over draws).

    Once continuity reaches a level, higher weights only reshuffle equally good schedules, so each line keeps
    one point per continuity level: the lowest weight that reaches it.
    """
    draws_total = sweep['seed'].nunique()
    points = (usable_runs(sweep).groupby(['nurse_count', 'w_c'])
              .agg(single_nurse_pct=('single_nurse_pct', 'mean'), total_cost=('total_cost', 'mean'),
                   draws=('seed', 'size'))
              .reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = sorted(sweep['nurse_count'].unique())
    for count, runs in points.groupby('nurse_count'):
        color = ROSTER_COLORS[sizes.index(count) % len(ROSTER_COLORS)]
        runs = (runs.assign(level=runs['single_nurse_pct'].round(1)).sort_values('w_c')
                .drop_duplicates('level', keep='first').sort_values('single_nurse_pct'))
        draws = int(runs['draws'].min())
        ax.plot(runs['single_nurse_pct'], runs['total_cost'], color=color, linewidth=2, marker='o', markersize=5,
                label=f'{count} nurses ({draws} of {draws_total} rosters)')
        end = runs.iloc[-1]
        ax.annotate(f'{count}', (end['single_nurse_pct'], end['total_cost']), xytext=(8, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=9, color=INK_2)

    style(ax, 'What continuity of care costs, by roster size',
          'Patients who keep one nurse all week (%)', 'Weekly route cost ($)')
    ax.margins(x=0.06)
    ax.legend(frameon=False, title='Roster size', loc='upper left')
    missing = sorted(set(sizes) - set(points['nurse_count']))
    if missing:
        fig.text(0.01, 0.01, f"Not shown: {', '.join(map(str, missing))} nurses - no roster drawn could cover the week. "
                 'Each line averages the rosters that give a drivable schedule at every weight.', fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, 'staffing_sweep')


def plot_cost_by_roster(sweep):
    """Weekly route cost per roster size, lowest-cost schedule vs full continuity, with the range over rosters."""
    draws_total = sweep['seed'].nunique()
    ok = usable_runs(sweep)
    sizes = sorted(sweep['nurse_count'].unique())
    policies = [(LOWEST_COST_W_C, 'Lowest cost (ignore continuity)', BLUE),
                (FULL_CONTINUITY_W_C, 'Full continuity (one nurse per patient)', ORANGE)]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.36
    top = ok['total_cost'].max() if len(ok) else 1
    for k, (w_c, label, color) in enumerate(policies):
        runs = ok[ok['w_c'] == w_c].groupby('nurse_count')['total_cost'].agg(['mean', 'min', 'max', 'size'])
        for n, count in enumerate(sizes):
            x = n + (k - 0.5) * width
            if count not in runs.index:
                continue
            r = runs.loc[count]
            ax.bar(x, r['mean'], width=width - 0.04, color=color, label=label,
                   edgecolor='white', linewidth=1)
            if r['size'] > 1:
                ax.errorbar(x, r['mean'], yerr=[[r['mean'] - r['min']], [r['max'] - r['mean']]],
                            color=INK_2, linewidth=1, capsize=3)
            ax.annotate(f"${r['mean']:.0f}", (x, r['max']), xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=INK_2)

    ticks = []
    for n, count in enumerate(sizes):
        feasible = ok[ok['nurse_count'] == count]['seed'].nunique()
        if feasible:
            ticks.append(f'{count} nurses\n{feasible} of {draws_total} rosters')
        else:
            ticks.append(f'{count} nurses\nnone')
            ax.text(n, top * 0.05, 'cannot cover\nthe week', ha='center', va='bottom', fontsize=8, color=INK_2)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc='upper left', ncol=2)
    ax.set_xticks(range(len(sizes)), ticks)
    ax.set_ylim(0, top * 1.22)
    style(ax, 'Weekly route cost by roster size', None, 'Weekly route cost ($)')
    ax.grid(axis='x', visible=False)
    fig.text(0.01, 0.01, 'Bars: average over the rosters that give a drivable schedule at every weight. '
             'Whiskers: cheapest to most expensive roster.', fontsize=8, color=INK_2)
    fig.tight_layout()
    save(fig, 'cost_by_roster')


# ---------------------------------------------------------------- route figures

def plot_nurse_time_breakdown(summary):
    """Each nurse's week: hours on visits, driving, and spare shift time (stacked to total shift hours)."""
    t = lambda s: s.map(seq.time_to_minutes)
    shift = t(summary['shift_end']) - t(summary['shift_start'])
    week = (summary.assign(shift=shift, drive=summary['route_drive_minutes'],
                           spare=summary['slack_min'])
            .assign(visit=lambda f: f['shift'] - f['drive'] - f['spare'])
            .groupby('nurse_id')[['visit', 'drive', 'spare', 'shift']].sum() / 60)
    week = week.sort_index(ascending=False) # N001 at the top

    fig, ax = plt.subplots(figsize=(8, 5.5))
    left = pd.Series(0.0, index=week.index)
    for col, label, color in [('visit', 'Visits', BLUE), ('drive', 'Driving', ORANGE), ('spare', 'Spare shift time', SPARE)]:
        ax.barh(week.index, week[col], left=left, color=color, label=label, edgecolor='white', linewidth=2, height=0.7)
        left += week[col]
    for nurse, row in week.iterrows():
        ax.text(row['shift'] + 0.4, nurse, f"{100 * (row['visit'] + row['drive']) / row['shift']:.0f}% busy",
                va='center', fontsize=8, color=INK_2)

    style(ax, 'How each nurse spends the week', 'Hours (Mon-Fri)')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, week['shift'].max() * 1.15)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    total = week[['visit', 'drive', 'shift']].sum()
    fig.text(0.01, 0.01, f"All nurses: {total['visit']:.0f} h visits, {total['drive']:.0f} h driving, "
             f"{100 * (total['visit'] + total['drive']) / total['shift']:.0f}% of shift hours in use.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'nurse_time_breakdown')


def plot_route_maps(summary, schedule, patients, nurses):
    """One figure per day: a small map per working nurse, her route drawn over every patient (gray)."""
    home = nurses.set_index('nurse_id')[['lat', 'lon']]
    loc = patients.set_index('patient_id')[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    lats = pd.concat([loc['lat'], home['lat']])
    lons = pd.concat([loc['lon'], home['lon']])
    pad_lat, pad_lon = (lats.max() - lats.min()) * 0.05, (lons.max() - lons.min()) * 0.05
    aspect = 1 / math.cos(math.radians(lats.mean())) # degrees of longitude are shorter than latitude here

    for day in seq.DAYS:
        routes = summary[summary['day'] == day].sort_values('nurse_id')
        if routes.empty:
            continue
        cols = 4
        rows = math.ceil(len(routes) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.1), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')

        for ax, route in zip(axes.flat, routes.itertuples()):
            stops = schedule[(schedule['nurse_id'] == route.nurse_id) & (schedule['day'] == day)].sort_values('stop')
            path = pd.concat([home.loc[[route.nurse_id]], loc.loc[stops['patient_id']], home.loc[[route.nurse_id]]])

            ax.scatter(loc['lon'], loc['lat'], s=4, color=SPARE, linewidths=0) # every patient, for context
            ax.plot(path['lon'], path['lat'], color=BLUE, linewidth=1.5, zorder=2)
            ax.scatter(path['lon'].iloc[1:-1], path['lat'].iloc[1:-1], s=55, color=BLUE,
                       edgecolors='white', linewidths=1.5, zorder=3)
            for n, (lon, lat) in enumerate(zip(path['lon'].iloc[1:-1], path['lat'].iloc[1:-1]), start=1):
                ax.text(lon, lat, str(n), ha='center', va='center', fontsize=6, color='white', zorder=4)
            ax.scatter(path['lon'].iloc[0], path['lat'].iloc[0], s=60, marker='s', color=INK, zorder=4)

            ax.set_xlim(lons.min() - pad_lon, lons.max() + pad_lon)
            ax.set_ylim(lats.min() - pad_lat, lats.max() + pad_lat)
            ax.set_aspect(aspect)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            visits = f"{route.stops} visit{'s' if route.stops != 1 else ''}"
            ax.set_title(f'{route.nurse_id}: {visits}, {route.route_miles:.0f} mi\n'
                         f'{route.shift_start} - back {route.return_time}', fontsize=8, color=INK, loc='left')

        fig.suptitle(f'{day}: nurse routes ({len(routes)} nurses, {routes["stops"].sum()} visits, '
                     f'{routes["route_miles"].sum():.0f} mi)', x=0.01, ha='left', color=INK, fontsize=12)
        fig.text(0.01, 0.005, 'Black square: nurse home (start and end). Numbered dots: visits in order. '
                 'Gray: all patients this week.', fontsize=8, color=INK_2)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        save(fig, f'route_map_{day}')


def web_mercator(lat, lon):
    """Lat/lon (degrees) to Web Mercator metres (EPSG:3857), the projection street tiles are drawn in."""
    r = 6378137.0
    x = lon.map(lambda v: r * math.radians(v))
    y = lat.map(lambda v: r * math.log(math.tan(math.pi / 4 + math.radians(v) / 2)))
    return x, y


def draw_background(ax):
    """Street tiles if contextily can fetch them, plus ZIP outlines if geopandas is installed.

    Returns the attribution the tiles require, or None if no tiles were drawn.
    """
    try:
        import geopandas as gpd
        zips = gpd.read_file(ZIP_BOUNDARIES).to_crs(epsg=3857)
        zips.boundary.plot(ax=ax, color=AXIS, linewidth=0.5, zorder=1)
    except Exception as e: # optional layer
        print(f'  (no ZIP outlines: {e})')

    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldGrayCanvas, attribution=False, zorder=0)
        return 'Basemap: Esri, HERE, Garmin, (c) OpenStreetMap contributors'
    except Exception as e: # no contextily, or no connection
        print(f'  (no street basemap: {e})')
        return None


def plot_route_map_all(summary, schedule, patients, nurses):
    """One figure per day: every nurse's route on one map, one colour per nurse, visits numbered in order.

    A nurse keeps the same colour on every day (colour follows the nurse, not who happens to work).
    """
    roster = sorted(nurses['nurse_id'])
    color = {n: NURSE_COLORS[k % len(NURSE_COLORS)] for k, n in enumerate(roster)}
    dashes = {n: '-' if k < len(NURSE_COLORS) else '--' for k, n in enumerate(roster)}
    marker = {n: 'o' if k < len(NURSE_COLORS) else 'D' for k, n in enumerate(roster)} # second cycle: diamonds too

    home = nurses.set_index('nurse_id')[['lat', 'lon']]
    home['x'], home['y'] = web_mercator(home['lat'], home['lon'])
    loc = patients.set_index('patient_id')[['latitude', 'longitude']]
    loc['x'], loc['y'] = web_mercator(loc['latitude'], loc['longitude'])

    # same extent every day, so the five maps line up
    xs, ys = pd.concat([loc['x'], home['x']]), pd.concat([loc['y'], home['y']])
    pad = 0.04 * max(xs.max() - xs.min(), ys.max() - ys.min())
    xlim, ylim = (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)
    halo = [pe.withStroke(linewidth=2.5, foreground='white')]

    for day in seq.DAYS:
        routes = summary[summary['day'] == day].sort_values('nurse_id')
        if routes.empty:
            continue
        fig, ax = plt.subplots(figsize=(14, 11 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))) # extra width: legend
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        attribution = draw_background(ax)

        handles = []
        for route in routes.itertuples():
            n, c = route.nurse_id, color[route.nurse_id]
            stops = schedule[(schedule['nurse_id'] == n) & (schedule['day'] == day)].sort_values('stop')
            path = pd.concat([home.loc[[n], ['x', 'y']], loc.loc[stops['patient_id'], ['x', 'y']],
                              home.loc[[n], ['x', 'y']]])

            ax.plot(path['x'], path['y'], color=c, linestyle=dashes[n], linewidth=2, zorder=3,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground='white')])
            ax.scatter(path['x'].iloc[1:-1], path['y'].iloc[1:-1], s=90, marker=marker[n], color=c, edgecolors='white',
                       linewidths=1.5, zorder=4)
            for k, (x, y) in enumerate(zip(path['x'].iloc[1:-1], path['y'].iloc[1:-1]), start=1):
                ax.text(x, y, str(k), ha='center', va='center', fontsize=6.5, fontweight='bold',
                        color='white', zorder=5, path_effects=[pe.withStroke(linewidth=1.2, foreground=c)])
            ax.scatter(path['x'].iloc[0], path['y'].iloc[0], s=110, marker='s', color=c, edgecolors=INK,
                       linewidths=1.2, zorder=6)
            ax.annotate(n, (path['x'].iloc[0], path['y'].iloc[0]), xytext=(7, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold', color=INK, zorder=7, path_effects=halo)

            visits = f"{route.stops} visit{'s' if route.stops != 1 else ''}"
            handles.append(Line2D([], [], color=c, linestyle=dashes[n], linewidth=2, marker=marker[n], markersize=7,
                                  markeredgecolor='white', label=f'{n}: {visits}, {route.route_miles:.0f} mi'))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        ax.legend(handles=handles, title='Nurse', loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False,
                  fontsize=9, title_fontsize=10, labelcolor=INK)
        ax.set_title(f'{day}: all nurse routes ({len(routes)} nurses, {routes["stops"].sum()} visits, '
                     f'{routes["route_miles"].sum():.0f} mi)', loc='left', color=INK, fontsize=13)
        note = ('Square: nurse home (start and end). Numbered markers: visits in order. '
                'Lines join stops directly; they are not the road path.')
        if attribution:
            note += f'  {attribution}'
        fig.text(0.01, 0.01, note, fontsize=8, color=INK_2)
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        save(fig, f'route_map_all_{day}')


# ---------------------------------------------------------------- main

def main():
    print('Saving figures to', FIGURES_DIR)
    if SWEEP.exists():
        sweep = pd.read_csv(SWEEP)
        plot_staffing_sweep(sweep)
        plot_cost_by_roster(sweep)
    else:
        print(f'  skipped sweep figures: {SWEEP} not found (run python -m Optimization_Model.sweep)')

    summary = pd.read_csv(ROUTE_SUMMARY)
    schedule = pd.read_csv(ROUTE_SCHEDULE)
    plot_nurse_time_breakdown(summary)
    patients, nurses = pd.read_csv(PATIENTS), pd.read_csv(NURSES)
    plot_route_maps(summary, schedule, patients, nurses)
    plot_route_map_all(summary, schedule, patients, nurses)


if __name__ == '__main__':
    main()