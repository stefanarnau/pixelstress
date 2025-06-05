# Imports
import os
import pandas as pd
import numpy as np
import seaborn as sns

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




























        