import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
data = pd.read_csv(
    "C:\\Users\\DMWeiy\\OneDrive\\Documents\\SET1_Water_Quality_1_Year.csv"
)
df = pd.DataFrame(data)
df = df.sort_values("Timestamp")

df["pH"] = df["pH"].ffill()
df["Temperature"] = df["Temperature"].bfill()
df["Ammonia"] = df["Ammonia"].ffill()

timestamp_groups = df.groupby('Timestamp')

# Create a list to store the grouped data
grouped_data = []
group_number = 1  # Initialize group number

# Get the maximum number of rows in any timestamp group (in this case, it's 3)
max_group_size = max(timestamp_groups.size())

# Iterate over the groups and create new groups by picking one row from each timestamp group
for i in range(max_group_size):
    group = []
    for timestamp, group_data in timestamp_groups:
        if i < len(group_data):  # Make sure there's a row to pick
            group_data_with_group = group_data.iloc[i].copy()  # Copy the row
            group_data_with_group['Group'] = group_number  # Add the group number
            group.append(group_data_with_group)  # Add one row from each timestamp group
    grouped_data.append(pd.concat(group, axis=1).T)  # Concatenate and transpose back to row format
    group_number += 1  # Increment the group number

# Combine all the groups into a single DataFrame
final_df = pd.concat(grouped_data, ignore_index=True)

group1_df = final_df[final_df['Group'] == 1]

# Remove non-numeric columns (Timestamp and Group)
group1_df = group1_df.drop(columns=['Timestamp', 'Group'])

# Plot heatmap
plt.figure(figsize=(8, 6))
sb.heatmap(group1_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Show the plot
plt.title('Heatmap for Group 1')
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(18, 6))

# Scatter plot of Attribute1 vs Attribute2
axes[0].scatter(group1_df['pH'], group1_df['Temperature'], color='blue', alpha=0.7)
axes[0].set_xlabel('pH value')
axes[0].set_ylabel('Temperature')
axes[0].set_title('pH value vs Temperature')

plt.tight_layout()

# Show the plot
plt.suptitle('Scatter Plots for Group 1', fontsize=16)
plt.show()

final_df.to_csv('E:\\Pycharm\\IML_Project\\New_Water_Quality_AfterCleaning.csv', index=False)
# Show the resulting DataFrame
print(final_df)