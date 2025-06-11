# Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Paths
path_in = "/home/plkn/repos/pixelstress/subjective_ratings/"
path_out = "/mnt/data_dump/pixelstress/4_results/"

# The file to read
fn = os.path.join(path_in, "ratings.csv")

# Load as df
df = pd.read_csv(fn)

# Remove columns
df = df.drop(['Gruppe', 'Filter'], axis=1)

# Rename columns
df = df.rename(columns={
    'Group': 'group',
    'Subject': 'id',
    'Geschlecht': 'sex',
    'Alter': 'age',
})

# Select columns 'A' and 'C' to create a new DataFrame
df_demo = df[['id', 'group', 'sex', 'age']]

# Remove columns
df = df.drop(['sex', 'age'], axis=1)

# Identify dv columns
dvcols = [col for col in df.columns if col not in ['id', 'group']]

# New rows
new_rows = []

# Iterate df
for idx, row in df.iterrows():
    
    # Iterate dvcols
    for dvcol in dvcols:
        
        # Get factors  
        feedback = dvcol.split('_')[0][:-1]
        stage = dvcol.split('_')[0][-1]
        
        # Rejoin dv label
        dv = '_'.join(dvcol.split('_')[1:])
        
        # Compile
        new_rows.append({'id':row['id'],
                         'group':row['group'],
                         'stage':stage,
                         'feedback':feedback,
                         'dv':dv,
                         'value': float(row[dvcol]),
                         })

# New df
df = pd.DataFrame(new_rows)
df = df.pivot(index=['id', 'group', 'stage', 'feedback'], columns='dv', values='value').reset_index()
        
# Rename columns
df = df.rename(columns={
    'Anstrengung': 'effort',
    'ERREGUNG': 'arousal',
    'Frustration': 'frustration',
    'Geistige_Anforderung': 'mental demand',
    'Leistung': 'performance',
    'STRESS': 'stress',
    'Zeitliche_Anforderung': 'temporal demand',

})

# Rename values
df['group'] = df['group'].replace({1: 'exp', 2: 'ctrl'}) 
df['stage'] = df['stage'].replace({'1': 'start', '2': 'end'}) 
df['feedback'] = df['feedback'].replace({'Close': 'close', 'Above': 'above', 'Below': 'below'})

# Make id category
df['id'] = df['id'].astype('category')

# Set seaborn params
lineplot_palette = ["#888888", "#bb11bb", "#11bbbb"]
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Identify dv columns
dvcols = [col for col in df.columns if col not in ['id', 'group', 'stage', 'feedback']]
    
# Save ratings to csv
fn = os.path.join(path_out, "stats_table_ratings.csv")
df.to_csv(fn, index=False)



# Melt the dependent variables into long format
df_long = df.melt(
    id_vars=["id", "group", "stage", "feedback"],
    value_vars=["effort", "arousal", "frustration", "mental demand", "performance", "stress", "temporal demand"],
    var_name="dependent_variable",
    value_name="value"
)

# =====================================

# Create a new column for the 12 combinations of within- and between-subject factors
df_long["combination"] = df_long["group"].astype(str) + "_" + df_long["stage"].astype(str) + "_" + df_long["feedback"].astype(str)

# Aggregate (e.g., mean over subjects for each combination and dependent variable)
summary = df_long.groupby(["combination", "dependent_variable"], as_index=False)["value"].mean()

# Sort combination levels to ensure consistent x-axis order
summary["combination"] = pd.Categorical(summary["combination"], categories=sorted(summary["combination"].unique()), ordered=True)


# Define style combinations for: arousal, effort, frustration, mental demand, performance, stress, temporal demand
colors = ['darkslategray', 'darkslategray', 'teal', 'teal', 'purple', 'purple', 'olive']
markers = ['o', 'D', 'o', 'D', 'o', 'D', 'o']
linestyles = ['-', ':', '-', ':', '-', ':', '-']

# Map dependent_variable to styles
unique_vars = summary['dependent_variable'].unique()
style_dict = {
    var: {
        "color": colors[i % len(colors)],
        "marker": markers[i % len(markers)],
        "linestyle": linestyles[i % len(linestyles)]
    } for i, var in enumerate(unique_vars)
}

# Set font sizes
plt.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.titlesize': 24,      # Title font
    'axes.labelsize': 18,      # X and Y label font
    'xtick.labelsize': 14,     # X tick labels
    'ytick.labelsize': 14,     # Y tick labels
    'legend.fontsize': 16,     # Legend font
    'legend.title_fontsize': 16  # Legend title font
})

# Plot
fig = plt.figure(figsize=(18, 8))

# Plot lines
for var in unique_vars:
    subset = summary[summary['dependent_variable'] == var]
    style = style_dict[var]
    plt.plot(
        subset['combination'],
        subset['value'],
        label=var,
        marker=style['marker'],
        linestyle=style['linestyle'],
        color=style['color'],
        linewidth=2.5,        
        markersize=8  
    )

plt.xticks(rotation=45)
plt.xlabel("Factor Combination")
plt.ylabel("Rating")
plt.title("Subjective ratings across experimental conditions")

# Adjust bottom margin
plt.subplots_adjust(bottom=0.2)  

# Move legend to bottom
plt.legend(
    title="Rating",
    bbox_to_anchor=(0.5, -0.33),
    loc="upper center",
    frameon=False,
    ncol=4
)

# Save
fn = os.path.join(path_out, "ratings.png")
fig.savefig(fn, dpi=300, bbox_inches='tight')



























        