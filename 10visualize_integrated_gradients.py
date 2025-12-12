import pandas as pd
import matplotlib.pyplot as plt

#read genral settings
basin = '02465493' # '06447500', '10336660', '02465493'
param_names = [ "fc", "beta", "pwp", "l", "ks", "ki", "kb", "kperc",
                "coeff_pet", "ddf", "scf", "ts", "tm", "tti", "whc",
                "crf", "d_shape", "d_scale", "b_shape", "b_scale"]

##-------------------------------------------------------------------##--------------------------------------------------------------------------##-----------------------------------------------------------
## -- Feature importance for a selected parameter
df_og = pd.read_csv(f'output/integrated_gradients/lstm_1hbv/ig_lstm_hbv_{basin}_par_{param_names[0]}.csv')
df_og['date'] = pd.to_datetime(df_og['date'])
df = df_og.drop(columns=['date', 'lag'])
df_mean_abs = df.abs().mean().sort_values(ascending=False) # mean of absolute values

#plot feature importance plot for all features
plt.figure(figsize=(8,5))
df_mean_abs.plot(kind='bar')
plt.title(f'Feature Importance for Basin {basin} - Parameter: {param_names[0]}')
plt.ylabel('Mean of Absolute Attribution Values')
plt.xlabel('Features')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(f'figures/integrated_gradients/feature_importance_{basin}_par_{param_names[0]}.png',dpi=300)
plt.show()


##-------------------------------------------------------------------##--------------------------------------------------------------------------##-----------------------------------------------------------
## Feature importance for all parameters combined
# merge all parameter IG dataframes
df_list = []
for par in param_names:
    df_par = pd.read_csv(f'output/integrated_gradients/lstm_1hbv/ig_lstm_hbv_{basin}_par_{par}.csv')
    df_par = df_par.drop(columns=['date', 'lag'])
    df_list.append(df_par)
df_merged = pd.concat(df_list, axis=0, ignore_index=True)
df_mean_abs_all = df_merged.abs().mean().sort_values(ascending=False) # mean of absolute values
#plot feature importance plot for all features and all parameters
plt.figure(figsize=(10,6))
df_mean_abs_all.plot(kind='bar')
plt.title(f'Feature Importance (Integrated Gradients) for Basin {basin} - All Parameters')
plt.ylabel('Mean of Absolute Attribution Values')
plt.xlabel('Features and Parameters')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(f'figures/integrated_gradients/feature_importance_{basin}_all_parameters.png',dpi=300)
plt.show()


##-------------------------------------------------------------------##--------------------------------------------------------------------------##-----------------------------------------------------------
##--- Time series of IG for as selected parameter for selected feature and selected data
date = '1991-04-30'
feature = 'precip'

# Read input file and get precipitation data
df_input = pd.read_csv(f'data/input_{basin}.csv')
df_input['date'] = pd.to_datetime(df_input['date'])
df_input = df_input[['date', 'precip']]
df_input_selected = df_input[df_input['date'] <= date].tail(365).reset_index(drop=True)

# Get IG data for selected date
df_time = df_og[(df_og['date'] == date)][['date', 'lag', feature]].reset_index(drop=True)

# Check lengths match
if len(df_time) != len(df_input_selected):
    raise ValueError('Length of IG data does not match length of input data for the selected date.')

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Plot precipitation on right y-axis (top subplot area)
ax2.plot(df_time['lag'], df_input_selected['precip'], 'orange', label='Precipitation', alpha=0.6)
ax2.set_ylabel('Precipitation (mm/day)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.invert_yaxis()  # Invert so it appears to hang from top

# Plot IG on left y-axis (bottom subplot area)
ax1.plot(df_time['lag'], df_time[feature], 'b-o', label='IG Attribution', markersize=4)
ax1.set_xlabel('Lag')
ax1.set_ylabel('Integrated Gradients Attribution', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Add top x-axis with dates
ax3 = ax1.twiny()
ax3.set_xlim(ax1.get_xlim())
ax3.set_xlabel('Date')
# Set date ticks at regular intervals
num_ticks = 6
tick_indices = [i * (len(df_input_selected) - 1) // (num_ticks - 1) for i in range(num_ticks)]
ax3.set_xticks([df_time['lag'].iloc[i] for i in tick_indices])
ax3.set_xticklabels([df_input_selected['date'].iloc[i].strftime('%Y-%m-%d') for i in tick_indices], rotation=0)

plt.title(f'Integrated gradients attribution for {feature} on {date} (Basin {basin})')
plt.tight_layout()
plt.savefig(f'figures/integrated_gradients/ig_timeseries_{basin}_par_{param_names[0]}_{feature}_{date}.png', dpi=300)
plt.show()


# ##-------------------------------------------------------------------##--------------------------------------------------------------------------##-----------------------------------------------------------
# # Time series of mean IG for a parameter for selected feature and selected date
# feature = 'precip'
# # date are all unique dates in the dataset which then would be averaged over
# df_dates = df_og[['date']].drop_duplicates().reset_index(drop=True)
# mean_ig_list = []
# for idx, row in df_dates.iterrows():
#     date = row['date']
#     df_time = df_og[(df_og['date'] == date)][['date', 'lag', feature]].reset_index(drop=True)
#     mean_ig = df_time[feature].mean()
#     mean_ig_list.append({'date': date, 'mean_ig': mean_ig})
# df_mean_ig = pd.DataFrame(mean_ig_list)
# df_mean_ig['date'] = pd.to_datetime(df_mean_ig['date'])
# # Plot time series of mean IG
# plt.figure(figsize=(10,5))
# plt.plot(df_mean_ig['date'], df_mean_ig['mean_ig'], marker='o')
# plt.xlabel('Date')
# plt.ylabel('Mean Integrated Gradients Attribution')
# plt.title(f'Time Series of Mean IG for Feature: {feature}')
# plt.xticks(rotation=45)
# plt.tight_layout()
# # plt.savefig(f'figures/integrated_gradients/mean_ig_timeseries_{basin}_feature_{feature}.png', dpi=300)
# plt.show()
