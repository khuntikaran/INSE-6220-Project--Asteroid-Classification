import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
file_path = 'data/processed_nasa_data.csv'
data = pd.read_csv(file_path)


# Select a subset of columns for the pair plot


# 1. Box Plot of Absolute Magnitude by Hazardous Category
def plot_absolute_magnitude_by_hazardous(data):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Hazardous', y='Absolute Magnitude', data=data)
    plt.title('Absolute Magnitude by Hazardous Category')
    plt.show()

# 2. Box Plot of Est Dia in KM(max) by Hazardous Category
def plot_est_dia_by_hazardous(data):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Hazardous', y='Est Dia in KM(max)', data=data)
    plt.title('Est Dia in KM(max) by Hazardous Category')
    plt.show()

# 3. Bar Chart of Mean Relative Velocity km per hr by Hazardous Category
def plot_relative_velocity_by_hazardous(data):
    plt.figure(figsize=(8, 6))
    data.groupby('Hazardous')['Relative Velocity km per hr'].mean().plot(kind='bar')
    plt.title('Mean Relative Velocity km per hr by Hazardous Category')
    plt.show()

# 4. Density Plot of Orbital Period by Hazardous Category
def plot_orbital_period_density(data):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data[data['Hazardous']]['Orbital Period'], label='Hazardous', shade=True)
    sns.kdeplot(data[~data['Hazardous']]['Orbital Period'], label='Non-Hazardous', shade=True)
    plt.title('Orbital Period Density by Hazardous Category')
    plt.legend()
    plt.show()

# 5. Scatter Plot of Miss Dist.(Astronomical) vs Relative Velocity km per sec by Hazardous Category
def plot_miss_dist_vs_relative_velocity(data):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Miss Dist.(Astronomical)', y='Relative Velocity km per sec', hue='Hazardous', data=data)
    plt.title('Miss Dist vs Relative Velocity by Hazardous Category')
    plt.show()

# 6. Histogram of Mean Motion by Hazardous Category
def plot_mean_motion_histogram(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data, x='Mean Motion', hue='Hazardous', element='step')
    plt.title('Histogram of Mean Motion by Hazardous Category')
    plt.show()

# Example calls for each function
plot_absolute_magnitude_by_hazardous(data)
plot_est_dia_by_hazardous(data)
plot_relative_velocity_by_hazardous(data)
plot_orbital_period_density(data)
plot_miss_dist_vs_relative_velocity(data)
plot_mean_motion_histogram(data)
