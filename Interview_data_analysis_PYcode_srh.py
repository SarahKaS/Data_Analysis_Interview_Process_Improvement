
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

events = pd.read_csv(r"C:\Users\sarah\OneDrive\Documents\DATA SCIENCE\events.csv")
sessions = pd.read_csv(r"C:\Users\sarah\OneDrive\Documents\DATA SCIENCE\sessions.csv")
applicants = pd.read_csv(r"C:\Users\sarah\OneDrive\Documents\DATA SCIENCE\applicants.csv")

# events.shape
# sessions.shape
# applicants.shape

print("\nShape of events tab with duplicate:")
print(events.shape)

# drop duplicate rows
events.drop_duplicates(keep='first', inplace=True)
sessions.drop_duplicates(keep='first', inplace=True)
applicants.drop_duplicates(keep='first', inplace=True)

# events.shape
# sessions.shape
# applicants.shape

print("\nShape of events tab without duplicate:")
print(events.shape)

events['event_datetime'].dtype

frames = [events, applicants, sessions]

# join tabs with 'session_id' and 'applicant_id' as common key
data = pd.merge(events,sessions, on=['session_id', 'applicant_id'], how='left')
data = pd.merge(data,applicants, on=['session_id', 'applicant_id'], how='left')

# Remove useless columns
data = data.drop(["first_name", 'last_name'], axis=1)
# create new dataframe with event_type: "Recruiter submitted test results" or "end_of_underwriting" not made by Recruiter

submission_df = data[(data["event_type"] == "Recruiter submitted test results") |
                     ((data["event_type"] == "end_of_underwriting") & (data["event_user"] != 'Recruiter'))]


submission_df.reset_index(inplace=True)

# check and fix data types
# submission_df.dtypes

submission = submission_df.copy()

submission.loc[:, 'event_datetime'] = pd.to_datetime(submission['event_datetime'])
submission.loc[:, 'birth_date'] = pd.to_datetime(submission['birth_date'])

submission = submission.sort_values(by=['session_id', 'event_datetime'])
submission.reset_index(inplace=True)


# Vectors of submission time before and after the modification in underwiting application on March 15, 2259:

submission_time_before = []
submission_time_after = []

modification_date = pd.Timestamp('2259-03-15', tz='UTC')

submission['delta_time'] = np.nan
submission['Before_After'] = np.nan
submission['event_datetime'].dtype

for i in range(len(submission) - 1):
    if (submission.iloc[i + 1]['event_type'] == 'Recruiter submitted test results' and
            submission.iloc[i]['event_type'] == 'end_of_underwriting' and
            submission.iloc[i + 1]['session_id'] == submission.iloc[i]['session_id']):
        delta_time = submission.iloc[i + 1]['event_datetime'] - submission.iloc[i]['event_datetime']
        if delta_time.seconds > 0 and delta_time.seconds < 7200:
            submission['delta_time'][i] = delta_time

            # before modification
            if submission.iloc[i]['event_datetime'] < modification_date:
                Before_After='Before'
                submission_time_before.append(delta_time)

            # after modification
            if submission.iloc[i]['event_datetime'] >= modification_date:
                Before_After = 'After'
                submission_time_after.append(delta_time)
            submission['Before_After'][i] = Before_After

# delete unrelevant rows:
submission = submission.dropna(subset=['Before_After'])

# reset the index after dropping rows
submission.reset_index(drop=True, inplace=True)



# Let's remove outliers to get a precise result.
# We can easily see that the process mostly take minutes, so values bigger than 2 hours will be removed. It will also drop negative results.

submission_time_before = [delta for delta in submission_time_before if delta.seconds < 7200]
submission_time_after = [delta for delta in submission_time_after if delta.seconds < 7200]

# Calculate the average
average_time_before = sum(submission_time_before, pd.Timedelta(0)) / len(submission_time_before)
average_time_after = sum(submission_time_after, pd.Timedelta(0)) / len(submission_time_after)

# Calculate the median
median_time_before = pd.Series(submission_time_before).median()
median_time_after = pd.Series(submission_time_after).median()


# Print the results

# print("Average Time before modification:", average_time_before)
# print("Average Time after modification:", average_time_after)
# print("Median Time before modification:", median_time_before)
# print("Median Time after modification:", median_time_after)



# in a more readable format (minutes:secondes):
# Convert average and median times to total seconds
average_sec_before = int(average_time_before.total_seconds())
median_sec_before = int(median_time_before.total_seconds())
average_sec_after = int(average_time_after.total_seconds())
median_sec_after = int(median_time_after.total_seconds())

# Format average and median times
formatted_avg_before = f"{(average_sec_before % 3600) // 60:02d}:{average_sec_before % 60:02d}"
formatted_avg_after = f"{(average_sec_after % 3600) // 60:02d}:{average_sec_after % 60:02d}"
formatted_median_before = f"{(median_sec_before % 3600) // 60:02d}:{median_sec_before % 60:02d}"
formatted_median_after = f"{(median_sec_after % 3600) // 60:02d}:{median_sec_after % 60:02d}"

# Print the results
print("Average Time before modification:", formatted_avg_before)
print("Average Time after modification:", formatted_avg_after)
print()
print("Median Time before modification:", formatted_median_before)
print("Median Time after modification:", formatted_median_after)

# Considering these results, we observe a significant increase in the average time after the change implementation.
# This suggests that the change might have had a negative impact.
# The median shows better results than the mean but is still twice higher than before the modification.


# Now we will try to understand if those bad results are related to some difference between datasets parameters before and after the modifications.


# Age of the applicants before and after:

# Ensure the datetime columns are in the correct format
submission['event_datetime'] = pd.to_datetime(submission['event_datetime']).dt.tz_localize(None)
submission['birth_date'] = pd.to_datetime(submission['birth_date']).dt.tz_localize(None)

submission['applicant_Age'] = (submission['event_datetime'] - submission['birth_date']).astype('<m8[Y]')
# print(submission)

# Calculate the average age before and after modification:
average_age_before = submission[submission['Before_After'] == 'Before']['applicant_Age'].mean()
average_age_after = submission[submission['Before_After'] == 'After']['applicant_Age'].mean()

print(f"Average age before the modification: {average_age_before:.2f} years")
print(f"Average age after  the modification: {average_age_after:.2f} years")

# We got a very close result before and after the modification (68.4 vs 67.7),
# so this parameter can't explain why the submission time increased after the modification.



# Now let's seee the submission time per gender:
# Filter the DataFrame for 'Before' and 'After'
before_df = submission[submission['Before_After'] == 'Before']
after_df = submission[submission['Before_After'] == 'After']

# Count the number of men and women for 'Before'
male_count_before = before_df[before_df['gender'] == 'male'].shape[0]
female_count_before = before_df[before_df['gender'] == 'female'].shape[0]

# Count the number of men and women for 'After'
male_count_after = after_df[after_df['gender'] == 'male'].shape[0]
female_count_after = after_df[after_df['gender'] == 'female'].shape[0]

print("\nGender counts before change:")
print(f"Male: {male_count_before}")
print(f"Female: {female_count_before}")

print("\nGender counts after change':")
print(f"Male: {male_count_after}")
print(f"Female: {female_count_after}")

# Here again, no significant difference in the gender distribution


# Now let's see if the difference per Recruiter:
# Calculate the average delta_time for each Recruiter_name for "Before"
average_delta_time_before = submission[submission['Before_After'] == 'Before'].groupby('Recruiter_name')['delta_time'].mean()

# Calculate the average delta_time for each Recruiter_name for "After"
average_delta_time_after = submission[submission['Before_After'] == 'After'].groupby('Recruiter_name')['delta_time'].mean()

print("\nAverage delta_time before by Recruiter_name:")
print(average_delta_time_before)

print("\nAverage delta_time after by Recruiter_name:")
print(average_delta_time_after)


# We can see it as a plot:
# Convert timedelta to total seconds for plotting
average_delta_time_before_seconds = average_delta_time_before.apply(lambda x: x.total_seconds())
average_delta_time_after_seconds = average_delta_time_after.apply(lambda x: x.total_seconds())

# Plotting the averages
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Plot for 'Before'
ax[0].bar(average_delta_time_before_seconds.index, average_delta_time_before_seconds.values, color='blue')
ax[0].set_title('Average Delta Time Before Modification per Recruiter Name')
ax[0].set_xlabel('Recruiter Name')
ax[0].set_ylabel('Average Delta Time (seconds)')
ax[0].tick_params(axis='x', rotation=90)

# Plot for 'After'
ax[1].bar(average_delta_time_after_seconds.index, average_delta_time_after_seconds.values, color='green')
ax[1].set_title('Average Delta Time After Modification per Recruiter Name')
ax[1].set_xlabel('Recruiter Name')
ax[1].set_ylabel('Average Delta Time (seconds)')
ax[1].tick_params(axis='x', rotation=90)

# Adjust layout
plt.tight_layout()


# Define the width of each bar
bar_width = 0.35

# Define the x positions for the bars
x_before = np.arange(len(average_delta_time_before_seconds))
x_after = np.arange(len(average_delta_time_after_seconds))

# Plotting the averages for both "Before" and "After" on separate plots
plt.figure(figsize=(12, 8))

# Plot for 'Before'
plt.bar(x_before, average_delta_time_before_seconds.values, width=bar_width, color='blue', label='Before')

# Plot for 'After'
plt.bar(x_after + bar_width, average_delta_time_after_seconds.values, width=bar_width, color='green', alpha=0.5, label='After')

plt.title('Average Delta Time Before and After Modification per Recruiter Name')
plt.xlabel('Recruiter Name')
plt.ylabel('Average Delta Time (seconds)')
plt.xticks(np.arange(max(len(x_before), len(x_after))), average_delta_time_before_seconds.index, rotation=90)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

