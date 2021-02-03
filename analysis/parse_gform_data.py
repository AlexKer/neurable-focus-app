import os.path as path
import pandas as pd
import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.express as px


##########
# Params #
##########
FP_GFORM_SHEET="/Users/alexker/Downloads/DogFooding (Responses) - Form Responses 1.tsv"

#! The published google form is only accessible to Neurable accounts
#! How would pandas provide crednetials?
#! Workaround - use browser to download labels and local file path
# GFORM_SHEET_URL="https://docs.google.com/spreadsheets/d/e/2PACX-1vSHuKKujv6Hv4j61r0wJ5dG5Ac6_dx4ZBoPQesd2GD7xPH7wDCxom7sJr-JwCuq1G7W_ncR8tXRasVl/pub?gid=911355979&single=true&output=tsv"
# lbl_df = pd.read_csv(GFORM_SHEET_URL, sep="\t")

# assert path.isfile(FP_GFORM_SHEET)


########
# Main #
########
# read and clean gform labels
lbl_df = pd.read_csv(FP_GFORM_SHEET, sep="\t")
lbl_df.columns = ['timestamp', 'activity', 'comments', 'recording']
lbl_df = lbl_df[['recording', 'timestamp', 'activity', 'comments']]
lbl_df['timestamp'] = pd.to_datetime(lbl_df['timestamp'])

# plot label timeseries
px.scatter(
    lbl_df, 
    x = "timestamp", 
    y = "recording", 
    color="activity", 
    hover_data=['recording', 'timestamp', 'activity', 'comments']
)