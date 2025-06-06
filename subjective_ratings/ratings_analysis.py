# Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Remove id==96 because missing data
df = df[df['id'] != 96]

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
df['group'] = df['group'].replace({'1': 'exp', '2': 'ctrl'}) 
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

# Iterate dvcols
for dvcol in dvcols:
        
    g = sns.catplot(
        data=df,
        x="group",
        y=dvcol,
        hue="feedback",
        kind="boxen",
        col="stage",
        k_depth=4,
        palette=lineplot_palette,
        col_order=["start", "end"],
    )
    
    # Save
    fn = os.path.join(path_out, dvcol + "_boxen.png")
    g.savefig(fn, dpi=300)
    
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

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=summary,
    x="combination",
    y="value",
    hue="dependent_variable",
    marker="o"
)

plt.xticks(rotation=45)
plt.xlabel("Factor Combination")
plt.ylabel("Dependent Variable Value")
plt.title("Lineplot of Dependent Variables Across Factor Combinations")
plt.tight_layout()
plt.show()































        