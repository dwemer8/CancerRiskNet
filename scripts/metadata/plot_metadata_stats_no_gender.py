"""
    This script generates some statistic on the data (Figure2 on the paper) 
"""
import pickle as pkl
import json
from collections import Counter
from tqdm import tqdm
import pickle
import numpy as np
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import os
from os.path import dirname, realpath
import torch
import argparse
import datetime
from dateutil.relativedelta import relativedelta
import sys
assert sys.version_info > (3,7,0)
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))
import cancerrisknet.datasets.factory as dataset_factory
import cancerrisknet.models.factory as model_factory
import cancerrisknet.learn.train as train
import cancerrisknet.learn.state_keeper as state
from cancerrisknet.utils.parsing import load_data_settings
from cancerrisknet.utils.parsing import md5, get_code
from cancerrisknet.utils.date import parse_date
from cancerrisknet.utils.visualization import save_figure_and_subplots


def print_stat(array, statistic='', denominator=None):
    array_count = Counter(array)
    if not denominator:
        normalized_array_count = {k:v/sum(array_count.values())*100 for k,v in array_count.items()}
    else:
        normalized_array_count = {k:v/denominator*100 for k,v in array_count.items()}
    print (f"###\t{statistic} stats")
    for k in sorted(array_count):
        print (f"{k}: {array_count[k]} ({normalized_array_count[k]:1.2f}%)")
    print ("##\n\n")

#generates two histograms side by side, one with birth as reference and the other as end of date/ pc diagnose
parser = argparse.ArgumentParser(description='Grid Search Results Collector.')
parser.add_argument('--save_dir', type=str, required=True, help="results dir")
parser.add_argument('--exp_id', type=str, required=True, help="exp id")
parser.add_argument('--data_setting_path', type=str, default='data/settings.yaml',
                    help="Path of yaml with data specific settings")
args = parser.parse_args()
data_settings = load_data_settings(args)
SETTINGS = data_settings['SETTINGS']
PANC_CANCER_CODE = SETTINGS.PANC_CANCER_CODE
results_path = os.path.join(args.save_dir, args.exp_id) + ".results"
resumed_args = pickle.load(open(results_path, "rb"))
args.__dict__ = resumed_args

print ("Summary statistics for metadata:\t{}".format(args.metadata_path))
j = json.load(open(args.metadata_path,'r'))

for p in j:
    j[p]['split_group']='train'
dataset_class = dataset_factory.get_dataset_class(args)
dataclass = dataset_class(j, args, 'train')
icd8_codes, icd10_codes = [], []
for p in tqdm(j): 
    for e in j[p]['events']: 
        code = get_code(args, e['codes'])
        if "ICD10_PANC_CANCER" in SETTINGS.__dict__ and \
            code in SETTINGS.ICD10_PANC_CANCER:
            icd10_codes.append(p) 
        elif "ICD8_PANC_CANCER" in SETTINGS.__dict__ and \
            code in SETTINGS.ICD8_PANC_CANCER:
            icd8_codes.append(p)

num_pc_icd10 = len(set(icd10_codes))
num_pc_icd8 = len(set(icd8_codes))
print (f"Original number of pancreatic cancer:\nICD10 codes {num_pc_icd10}\nICD8 codes {num_pc_icd8}")

icd8_codes, icd10_codes = [], []
for p in tqdm(dataclass.patients):
    if p['future_panc_cancer']:
        for e in p['events']:
            code = get_code(args, e['codes'])
            if "ICD10_PANC_CANCER" in SETTINGS.__dict__ and \
                code in SETTINGS.ICD10_PANC_CANCER:
                icd10_codes.append(p['patient_id']) 
            elif "ICD8_PANC_CANCER" in SETTINGS.__dict__ and \
                code in SETTINGS.ICD8_PANC_CANCER:
                icd8_codes.append(p['patient_id'])

num_pc_icd10 = len(set(icd10_codes))
num_pc_icd8 = len(set(icd8_codes))

print (f"After filtering number of pancreatic cancer:\nICD10 codes {num_pc_icd10}\nICD8 codes {num_pc_icd8}")
print ("Total number of patients:{}".format(len(j)))

valid_pids = [p for p in j if 'end_of_data' in j[p]]
status = ["Alive" if j[p]['end_of_data']== SETTINGS.END_OF_TIME_DATE.strftime("%Y-%m-%d") else "Dead" for p in valid_pids]
print_stat(status, statistic='Status All', denominator=len(valid_pids))

num_codes = {"all":[], 'cancer':[], "Both":[]}
timespan_trajectory = {"all":[], 'cancer':[]}
timepoint_codes = {'cancer':[]}
bins = np.array([-12*30,-6*30,-3*30, 0])
intervals = [
    "<12",
    "12-6",
    "6-3",
    "3-0",
]

for p in tqdm(dataclass.patients):
    age = relativedelta(p['outcome_date'], p['dob']).years
    age = (age//10)*10
    age_interval = '{}-{}'.format(age, age+10)
    trajectory_length = len([True for e in p['events'] if e['admit_date']<p['outcome_date']])
    num_codes['all'].append(trajectory_length)
    time_trajectory = relativedelta(p['outcome_date'], p['events'][0]['admit_date']).years
    timespan_trajectory['all'].append(time_trajectory)
    if p['future_panc_cancer']:
        for e in p['events']:
            timediff = (e['admit_date'] - p['outcome_date']).days
            if timediff>=0:
                continue
            timepoint_codes['cancer'].append(intervals[np.digitize(timediff, bins)])
        num_codes['cancer'].append(trajectory_length)
        timespan_trajectory['cancer'].append(time_trajectory)

print (f"### Total number of valid positive patients for the model:{sum([p['future_panc_cancer'] for p in dataclass.patients])}")

print_stat(timepoint_codes["cancer"], statistic='Number of codes at different bins Cancer Population')

print (f"Number of codes before outcome date for Processed Population: \
Mean {np.mean(num_codes['all'])} Median {np.median(num_codes['all'])} STD {np.std(num_codes['all'])}")
print (f"Number of codes before outcome date for Processed Cancer Population: \
Mean {np.mean(num_codes['cancer'])} Median {np.median(num_codes['cancer'])} STD {np.std(num_codes['cancer'])}")
print()
print (f"Length of trajectory for Processed Population: \
Mean {np.mean(timespan_trajectory['all'])} Median {np.median(timespan_trajectory['all'])} STD {np.std(timespan_trajectory['all'])}")
print (f"Length of trajectory for Processed Cancer Population: \
Mean {np.mean(timespan_trajectory['cancer'])} Median {np.median(timespan_trajectory['cancer'])} STD {np.std(timespan_trajectory['cancer'])}")
print()

#####
histograms_code_description = SETTINGS.DISEASE_HISTOGRAM
records_time_visit_patient_count = []
records_age_at_cancer = []
records_age_at_disease = []
records_disease_incidence = {bd:[0,0] for bd in histograms_code_description.values()}
is_cancer = [p['future_panc_cancer'] for p in dataclass.patients]
total_cancer_pt = sum(is_cancer)
total_non_cancer_pt = len(is_cancer) - total_cancer_pt

for p in tqdm(dataclass.patients):
    is_pc = p['future_panc_cancer']
    if is_pc:
        date_limit = p['outcome_date']
        age_at_cancer = relativedelta(p['outcome_date'], p['dob']).years
        records_age_at_cancer.append(age_at_cancer)
    else:
        date_limit = p['outcome_date'] - relativedelta(years=2)
    for e in p['events']:
        year = e['admit_date'].year
        records_time_visit_patient_count.append((year, is_pc))
        age = relativedelta(e['admit_date'], p['dob']).years
        records_age_at_disease.append((age, is_pc))
        code = get_code(args, e['codes'])
        if code in histograms_code_description and e['admit_date'] < date_limit:
            if is_pc:
                records_disease_incidence[histograms_code_description[code]][1]+=1
            else:
                records_disease_incidence[histograms_code_description[code]][0]+=1

export_pkl = [records_age_at_cancer, records_time_visit_patient_count, records_time_visit_patient_count, records_age_at_disease, records_disease_incidence]
pkl.dump(export_pkl, open("figures/metadata_stats_data.pkl", 'wb'))

plt.style.use('ggplot')
plt.rc('font', size=12)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=24)
plt.close()

fig = plt.figure(figsize=[24,7])

fig.suptitle('Dataset composition', fontsize=16)
gs = GridSpec(1,3)

ax1=fig.add_subplot(gs[0])
print ("records_per_year")
df = pd.DataFrame.from_records(records_time_visit_patient_count, columns=['Year', 'Cancer'])
ca = df[df.Cancer==True].Year.values
nonca = df[df.Cancer==False].Year.values
ax1.hist([ca, nonca], bins=40, label=["Yes", "No"], density=True, color=sns.color_palette(['#D4624E','#bebebe']))
ax1.set(xlim=(1990, 2021), xlabel='Year', ylabel='Disease code frequency', title="A - Disease distribution")
ax1.legend(title="PC")

ax1=fig.add_subplot(gs[1])
print ("age_at_disease_by_cancer")
df = pd.DataFrame.from_records(records_age_at_disease, columns=['Age', 'Cancer'])
ca = df[df.Cancer==True].Age.values
nonca = df[df.Cancer==False].Age.values
ax1.hist([ca, nonca], bins=52, label=["Yes", "No"], density=True, color=sns.color_palette(['#D4624E','#bebebe']))
ax1.legend(title="PC")
ax1.set(xlim=(0,100), xlabel='Age', ylabel='Disease code frequency', title='B - Disease distribution at age of PC')

ax1=fig.add_subplot(gs[2])
print ("disease_freq_by_cancer")
df = pd.DataFrame.from_dict(records_disease_incidence, columns=['No', 'Yes'], orient='index')
df = df.sort_values('Yes', ascending=False)
df = df[['Yes', 'No']]
df['disease_char'] = [l for i,l in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ') if i< len(df)] 

df = df.reset_index().rename(columns={'index':'disease'})
char_to_disease = dict(zip(df.disease_char, df.disease))
df['Yes'] = df['Yes']/total_cancer_pt
df['No'] = df['No']/total_non_cancer_pt
df = df.melt(id_vars=['disease', 'disease_char'])
sns.set_palette(sns.color_palette(['#D4624E','#bebebe']))
sns.barplot(data=df, x='disease_char', y='value', hue='variable', ax=ax1, alpha=0.8)
ax1.tick_params(axis='x', labelrotation=0)
ax1.legend(title="PC")
ax1.set(xlabel='Diseases', ylabel='Incidence', title="C - Prior knowledge disease incidence")
for label in ax1.get_yticklabels():
    label.set_ha('right')
cancer_legend = ax1.get_legend()
legend_patches = [mpatches.Patch(color='w', label=f"{k} : {v}") for k,v in char_to_disease.items()]
disease_legend = plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0,-0.2))
ax1.add_artist(disease_legend)
ax1.add_artist(cancer_legend)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
save_figure_and_subplots('figures/metadata_stats', fig, format='png', dpi=300)

df_c = pd.DataFrame(list(zip(num_codes['cancer'], timespan_trajectory['cancer'])), columns=['ncodes', 'trajlength'])
df_c['cancer'] = "yes"
df_a = pd.DataFrame(list(zip(num_codes['all'], timespan_trajectory['all'])), columns=['ncodes', 'trajlength'])
df_a['cancer'] = "no"
df = pd.concat([df_c, df_a])
df.to_csv("figures/metadata_stats_trajectory_length_all.csv")

x, y, hue = df['trajlength'], df['ncodes'], df['cancer']
jointdf = pd.DataFrame.from_records(zip(x,y,hue), columns=['Trajectory time range','# of codes', 'Cancer'])
xrange = range(0,41,4)
yrange = range(0,2001,100)

plt.subplots(figsize=[6,4])
ax = plt.subplot(121)
sns.histplot(data=jointdf.loc[jointdf['Cancer'] == 'no'], x='Trajectory time range', y='# of codes', bins=[xrange, yrange], pmax=0.5,
             ax=ax, element='step', cbar=True, cmap='mako_r', edgecolor="0.6", linewidth=0.4)
plt.yticks(yrange[::5])
plt.ylim([0, 2000]); plt.xlim([0, 40])

ax = plt.subplot(122)
sns.histplot(data=jointdf.loc[jointdf['Cancer'] == 'yes'], x='Trajectory time range', y='# of codes', bins=[xrange, yrange], pmax=0.5,
             ax=ax, element='step', cbar=True, cmap='rocket_r', edgecolor="0.6", linewidth=0.4)
plt.yticks(yrange[::5])
plt.ylim([0, 2000]); plt.xlim([0, 40])

plt.tight_layout()
plt.savefig('figures/metadata_stats_trajectory_length_heatmap.png', dpi=300)

df_a_sub = df_a.iloc[np.random.randint(0, df_a.shape[0], df_c.shape[0])]
df = pd.concat([df_c, df_a_sub])
df.to_csv("figures/metadata_stats_trajectory_length.csv")

plt.title('Trajectory codes vs. length')
sns.set_palette(sns.color_palette(['#D4624E','#bebebe']))

# Create figure and gridspec
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

# Main scatter plot
ax_main = fig.add_subplot(gs[1, 0])
jointdf = pd.DataFrame.from_records(zip(df['trajlength'], df['ncodes'], df['cancer']), 
                                  columns=['x','y','Cancer'])
sns.kdeplot(data=jointdf, x='x', y='y', hue='Cancer', bw_adjust=0.5, thresh=0.1, alpha=.6, ax=ax_main)
ax_main.set(xlim=(0, 40), ylim=(0, 2000))

# Top marginal (x-axis)
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
sns.histplot(data=jointdf, x='x', hue='Cancer', bins=40, stat='density', legend=False, alpha=.6, ax=ax_top)
ax_top.set(ylabel='Density')

# Right marginal (y-axis)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
sns.histplot(data=jointdf, y='y', hue='Cancer', bins=40, stat='density', legend=False, alpha=.6, ax=ax_right)
ax_right.set(xlabel='Density')

plt.tight_layout()
plt.savefig('figures/metadata_stats_trajectory_length.png', dpi=300)
