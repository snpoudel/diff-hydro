import pandas as pd
import matplotlib.pyplot as plt

#read ig csv file
basin = '01022500'
df_og = pd.read_csv(f'output/integrated_gradients/lstm_1hbv/ig_lstm_hbv_{basin}_par_0.csv')
df_og['date'] = pd.to_datetime(df_og['date'])
# read parquet file
# df = pd.read_parquet(f'Z:/tufts-hydro/regulation/output/integrated_gradients/{model}/ig_{basin}.parquet')

## -- Feature importance
df = df_og.drop(columns=['date', 'lag'])
df_mean_abs = df.abs().mean().sort_values(ascending=False) # mean of absolute values

#plot feature importance plot for all features
plt.figure(figsize=(8,5))
df_mean_abs.plot(kind='bar')
plt.title(f'Feature Importance (Integrated Gradients) for Basin {basin}')
plt.ylabel('Mean of Absolute Attribution Values')
plt.xlabel('Features')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
# plt.savefig(f'output/integrated_gradients/lstm_1hbv/feature_importance_{basin}.png',dpi=300)
plt.show()


##--- Time series of IG for selected feature and selected data
date = '2007-10-27'
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
plt.show()