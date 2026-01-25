import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide", page_title="Lake Malawi Land Use & ET Analysis")

st.title("RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin")
st.markdown("""
This dashboard displays the exploratory data analysis for the research on sustainable land-use allocation 
in the Lake Malawi Basin. It visualizes the relationship between land cover changes 
and evapotranspiration (ET) levels over the years 2017-2024.
""")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Paths relative to the root where app.py is located
    # Assumes app.py is in cs8903/ and data is in cs8903/data/
    lc_path = os.path.join("data", "processed", "land_cover_stats.json")
    et_path = os.path.join("data", "processed", "et_stats.json")
    
    # Load Land Cover Data
    with open(lc_path, 'r') as f:
        land_cover_stats = json.load(f)
        
    lc_data = []
    color_map = {}
    
    for year, data in land_cover_stats.items():
        row = {'Year': int(year)}
        for class_id, class_data in data['classes'].items():
            row[class_data['label']] = class_data['percentage']
            if class_data['label'] not in color_map:
                 color_map[class_data['label']] = class_data['color']
        lc_data.append(row)
        
    lc_df = pd.DataFrame(lc_data).sort_values('Year').set_index('Year')
    lc_df = lc_df.fillna(0)
    
    # Load ET Data
    with open(et_path, 'r') as f:
        et_stats_data = json.load(f)
        
    et_years = sorted([int(y) for y in et_stats_data.keys()])
    et_means = [et_stats_data[str(y)]['mean'] for y in et_years]
    et_p25 = [et_stats_data[str(y)]['p25'] for y in et_years]
    et_p75 = [et_stats_data[str(y)]['p75'] for y in et_years]
    
    et_df = pd.DataFrame({
        'Year': et_years, 
        'Mean_ET': et_means,
        'P25_ET': et_p25,
        'P75_ET': et_p75
    }).set_index('Year')
    
    return lc_df, et_df, color_map

# Attempt to load data
try:
    if os.path.exists(os.path.join("data", "processed", "land_cover_stats.json")):
        lc_df, et_df, color_map = load_data()
        st.success("Data loaded successfully!")
    else:
        st.error("Data files not found. Please ensure 'data/processed/' contains 'land_cover_stats.json' and 'et_stats.json'.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# --- Plot 1: Land Cover Composition ---
st.header("1. Land Cover Composition (2017-2024)")

# Re-create the plot logic from notebook
columns = [c for c in lc_df.columns if c in color_map]
colors = [color_map[c] for c in columns]

fig1, ax = plt.subplots(figsize=(10, 6))
lc_df[columns].plot(kind='bar', stacked=True, color=colors, width=0.8, ax=ax)

ax.set_title('Land Cover Composition', fontsize=14)
ax.set_ylabel('Percentage Coverage (%)', fontsize=10)
ax.set_xlabel('Year', fontsize=10)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Land Cover Class")
plt.tight_layout()

st.pyplot(fig1)

# --- Plot 2: ET vs Vegetation ---
st.header("2. Evapotranspiration vs. Vegetated Land Cover")

# Combine data
combined = lc_df.join(et_df, how='inner')
veg_cols = [c for c in ['Crops', 'Trees', 'Rangeland'] if c in combined.columns]
if veg_cols:
    combined['Vegetated Land'] = combined[veg_cols].sum(axis=1)

fig2, ax1 = plt.subplots(figsize=(12, 8))

# Function to calculate padded limits
def get_padded_limits(min_val, max_val, padding_factor=0.1):
    rng = max_val - min_val
    if rng == 0: rng = abs(min_val) * 0.1 if min_val != 0 else 1.0
    return min_val - (rng * padding_factor), max_val + (rng * padding_factor)

# Plot mean ET (Left Axis)
color_et = 'tab:blue'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Mean Evapotranspiration (kg/m²/year)', color=color_et, fontsize=12)

fill_1 = ax1.fill_between(combined.index, combined['P25_ET'], combined['P75_ET'], 
                 alpha=0.3, color=color_et, label='25th-75th Percentile ET')
line1 = ax1.plot(combined.index, combined['Mean_ET'], color=color_et, marker='o', linewidth=3, label='Mean ET')

ax1.tick_params(axis='y', labelcolor=color_et)
ax1.grid(True, alpha=0.3)

et_min_val = combined['P25_ET'].min()
et_max_val = combined['P75_ET'].max()
et_min_limit, et_max_limit = get_padded_limits(et_min_val, et_max_val, padding_factor=0.2)
ax1.set_ylim(et_min_limit, et_max_limit)

# Plot Land Cover (Right Axis)
ax2 = ax1.twinx()
lines = line1 + [fill_1] 
plot_lc_cols = []

if 'Crops' in combined.columns:
    color_crops = color_map.get('Crops', 'orange')
    line_crops = ax2.plot(combined.index, combined['Crops'], color=color_crops, marker='s', linestyle='--', linewidth=1.5, alpha=0.7, label='Crops %')
    lines += line_crops
    plot_lc_cols.append('Crops')

if 'Trees' in combined.columns:
    color_trees = color_map.get('Trees', 'green')
    line_trees = ax2.plot(combined.index, combined['Trees'], color=color_trees, marker='^', linestyle='--', linewidth=1.5, alpha=0.7, label='Trees %')
    lines += line_trees
    plot_lc_cols.append('Trees')

if 'Rangeland' in combined.columns:
    color_range = color_map.get('Rangeland', '#e3e2c3') 
    line_range = ax2.plot(combined.index, combined['Rangeland'], color=color_range, marker='v', linestyle='--', linewidth=1.5, alpha=0.7, label='Rangeland %')
    lines += line_range
    plot_lc_cols.append('Rangeland')

if 'Vegetated Land' in combined.columns:
    line_veg = ax2.plot(combined.index, combined['Vegetated Land'], color='darkgreen', marker='D', linestyle='-', linewidth=3, label='Total Vegetated Land %')
    lines += line_veg
    plot_lc_cols.append('Vegetated Land')

if plot_lc_cols:
    all_lc_values = combined[plot_lc_cols].values
    lc_min_val = all_lc_values.min()
    lc_max_val = all_lc_values.max()
    lc_min_limit, lc_max_limit = get_padded_limits(lc_min_val, lc_max_val, padding_factor=0.5) 
    lc_min_limit = max(0, lc_min_limit) if lc_min_val >= 0 else lc_min_limit
    ax2.set_ylim(lc_min_limit, lc_max_limit)

ax2.set_ylabel('Land Cover Percentage (%)', color='black', fontsize=12)
ax2.tick_params(axis='y', labelcolor='black')

labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

fig2.tight_layout()
st.pyplot(fig2)

st.markdown("""
### Conclusion
Basically, the mean **ET** has gradually decreased from 2017 to 2024, with some fluctuations during 2019 to 2022. This trend shows a **negative correlation** with the gradual increase of crop land and a **positive correlation** with the gradual decrease of total vegetated land (a sum of crops, trees, and rangeland).

There is also an obvious opposite change between trees and rangeland, which indicates a **potential substitution** for each other during the development period.

*Note: We did not calculate exact statistics like Pearson correlation as there are only 8 annual data points, which is too few to compare.*
""")
