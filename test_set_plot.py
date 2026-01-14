import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Category': ['BEV Easy', 'BEV Easy', 'BEV Easy', 'BEV Moderate', 'BEV Moderate', 'BEV Moderate', 'BEV Hard', 'BEV Hard', 'BEV Hard', '3D Easy', '3D Easy', '3D Easy', '3D Moderate', '3D Moderate', '3D Moderate', '3D Hard', '3D Hard', '3D Hard'],
    'Subcategory': ['Ours + 0% GT', 'Ours + 10% GT', 'Fully-supervised', 'Ours + 0% GT', 'Ours + 10% GT', 'Fully-supervised', 'Ours + 0% GT', 'Ours + 10% GT', 'Fully-supervised', 'Ours + 0% GT', 'Ours + 10% GT', 'Fully-supervised', 'Ours + 0% GT', 'Ours + 10% GT', 'Fully-supervised', 'Ours + 0% GT', 'Ours + 10% GT', 'Fully-supervised'],
    'Value': [90.09, 90.27, 90.44, 88.25, 87.78, 88.39, 86.95, 86.10, 87.86, 85.92, 88.95, 89.42, 75.33, 78.58, 84.06, 73.74, 77.42, 78.77]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set the plot style
sns.set(style="whitegrid")

# Create a grouped bar plot
plt.figure(figsize=(12, 3))
sns.barplot(x='Category', y='Value', hue='Subcategory', data=df, palette="icefire")

# Adding labels and title
plt.xlabel('Difficulty', fontsize=14)
plt.ylabel('mAP', fontsize=14)
plt.title('KITTI validation set 0.7 IoU', fontsize=16)

min_value = df['Value'].min()
max_value = df['Value'].max()
plt.ylim(min_value - 5, max_value + 5)

plt.savefig('point_plot.png', dpi=600, bbox_inches='tight')
# Show the plot
plt.show()
