import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import itertools

os.chdir("C:/Users/Zachary/OneDrive/Documents/Coding Projects/AI")

# Load your data (replace with the path to your dataset)
data = pd.read_csv('active22.csv')
data = data.drop(['INCLUDE', 'FIBER','MALE','FEMALE'], axis=1)
data = data.dropna()

# Sort the data by year (replace 'year_column' with the name of your year column)
data = data.sort_values('YEAR')
data = pd.get_dummies(data, columns=['VARIETY', 'LOC', 'STAGE'])
# List of traits (replace with your trait column names)
traits = ['T_SPACRE','TRS_TON', 'TCA', 'MSTWT', 'POPN']

window_size = 5
num_windows = data['YEAR'].nunique() - window_size

# Create an empty DataFrame to store the RMSE values
rmse_df = pd.DataFrame(columns=['Trait', 'Window', 'Stage Combination', 'RMSE'])

stages = ['OUTFIELD', 'INFIELD', 'NURSERY']

# Get all possible combinations of the stages
stage_combinations = []
for r in range(1, len(stages) + 1):
    stage_combinations += list(itertools.combinations(stages, r))

for trait in traits:
    print(f"Processing {trait}")

    for i in range(num_windows):
        for stage_comb in stage_combinations:
            print(f"Processing stage combination {stage_comb}")

            # Select data for the current window
            window_data = data[(data['YEAR'] >= data['YEAR'].unique()[i]) &
                               (data['YEAR'] < data['YEAR'].unique()[i + window_size])]

            # Filter the data based on the current stage combination
            stage_filter = [window_data[f"STAGE_{stage}"] == 1 for stage in stage_comb]
            stage_filter = np.logical_or.reduce(stage_filter)
            window_data = window_data[stage_filter]


            # Preprocess the data
            X = window_data.drop(traits, axis=1)
            y = window_data[trait]

            # Split the data into training, validation, and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

            # Normalize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Define the neural network model
            model = Sequential([
                Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')
            ])

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

            # Train the model
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=300)

            # Evaluate the model
            test_loss, test_mse = model.evaluate(X_test, y_test)
            test_rmse = np.sqrt(test_mse)
            print(f'Test MSE for window {i + 1} ({trait}, {stage_comb}): {test_mse}')
            print(f'Test RMSE for window {i + 1} ({trait}, {stage_comb}): {test_rmse}')
            rmse_df = rmse_df.append({'Trait': trait, 'Window': i + 1, 'Stage Combination': stage_comb, 'RMSE': test_rmse}, ignore_index=True)

rmse_df.to_csv('rmse_results_please.csv', index=False)
# Remove windows after 34 from rmse_df
rmse_df_filtered = rmse_df[rmse_df['Window'] <= 34]

# Group the rmse_df DataFrame by trait, window, and stage combination
grouped_rmse = rmse_df_filtered.groupby(['Trait', 'Window', 'Stage Combination'])

# Calculate the mean, minimum, maximum, and standard deviation for each group
stats_df = grouped_rmse.agg({'RMSE': ['mean', 'min', 'max', 'std']}).reset_index()

# Save the statistics to a CSV file
stats_df.to_csv('stats_by_window_stage_combination_trait.csv', index=False)


# Pivot the rmse_df DataFrame to have the 'Window' column as the main column
pivoted_rmse = rmse_df_filtered.pivot_table(index=['Window'], columns=['Trait', 'Stage Combination'], values='RMSE').reset_index()




# Compute the correlation matrix for the pivoted_rmse DataFrame
correlation_matrix = pivoted_rmse.corr()



# Create an empty DataFrame to store the variance values
variance_df = pd.DataFrame(columns=['Trait', 'Window', 'Stage Combination', 'Variance'])


for trait in traits:
    print(f"Processing {trait}")

    for i in range(num_windows):
        for stage_comb in stage_combinations:
            print(f"Processing stage combination {stage_comb}")

            # Select data for the current window
            window_data = data[(data['YEAR'] >= data['YEAR'].unique()[i]) &
                               (data['YEAR'] < data['YEAR'].unique()[i + window_size])]

            # Filter the data based on the current stage combination
            stage_filter = [window_data[f"STAGE_{stage}"] == 1 for stage in stage_comb]
            stage_filter = np.logical_or.reduce(stage_filter)
            window_data = window_data[stage_filter]

            # Compute the variance for the current window, stage combination, and trait
            variance = window_data[trait].var()

            # Add the computed variance to the variance_df DataFrame
            variance_df = variance_df.append({'Trait': trait, 'Window': i + 1, 'Stage Combination': stage_comb, 'Variance': variance}, ignore_index=True)


# Remove windows after 34 from variance_df
variance_df_filtered = variance_df[variance_df['Window'] <= 34]
# Pivot the variance_df DataFrame to have the 'Window' column as the main column
pivoted_variance = variance_df_filtered.pivot_table(index=['Window'], columns=['Trait', 'Stage Combination'], values='Variance').reset_index()


# Compute the correlation matrix for the pivoted_variance DataFrame
correlation_matrix_pivoted_variance = pivoted_variance.corr()

# Compute the correlation matrices for the pivoted_rmse and pivoted_variance DataFrames
correlation_matrix_pivoted_rmse = pivoted_rmse.corr()
correlation_matrix_pivoted_variance = pivoted_variance.corr()

# Subtract one correlation matrix from the other element-wise
matrix_difference = correlation_matrix_pivoted_rmse - correlation_matrix_pivoted_variance

# Compute the absolute differences
absolute_difference = matrix_difference.abs()



# Visualize the absolute differences using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(absolute_difference, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Absolute differences between correlation matrices")
plt.show()


# Subtract one correlation matrix from the other element-wise
matrix_difference = correlation_matrix_pivoted_rmse - correlation_matrix_pivoted_variance

# Compute the absolute differences
absolute_difference = matrix_difference.abs()

# Visualize the absolute differences using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(absolute_difference, annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Absolute differences between correlation matrices")
plt.show()


merged_df = pd.merge(rmse_df_filtered, variance_df_filtered, on=['Trait', 'Window', 'Stage Combination'], suffixes=('_RMSE', '_Variance'))

merged_df['RMSE_to_Variance_Ratio'] = merged_df['RMSE'] / merged_df['Variance']

# Create a copy of merged_df
merged_df_copy = merged_df.copy()

# Group by 'Trait' and 'Stage Combination', and calculate the mean for each group
mean_results = merged_df_copy.groupby(['Trait', 'Stage Combination']).mean().reset_index()


import seaborn as sns
import matplotlib.pyplot as plt

# Iterate through the unique traits in the mean_results DataFrame
for trait in mean_results['Trait'].unique():
    # Create a DataFrame with data for the current trait
    trait_data = mean_results[mean_results['Trait'] == trait]
    
    # Sort the data by RMSE-to-variance ratio in ascending order (lower values first)
    trait_data = trait_data.sort_values('RMSE_to_Variance_Ratio')
    
    # Create a bar plot for the current trait
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Stage Combination', y='RMSE_to_Variance_Ratio', data=trait_data)
    
    # Set the title and labels
    plt.title(f'RMSE to Variance Ratio for {trait}')
    plt.xlabel('Stage Combination')
    plt.ylabel('RMSE to Variance Ratio')
    
    # Rotate the x-axis labels to prevent overlapping
    plt.xticks(rotation=45, ha='right')
    
    # Show the plot
    plt.show()

# Create a copy of merged_df
merged_df_copy2 = merged_df.copy()

# Group by 'Trait', 'Window', and 'Stage Combination', and calculate the mean for each group
mean_results2 = merged_df_copy2.groupby(['Trait', 'Window', 'Stage Combination']).mean().reset_index()

import seaborn as sns
import matplotlib.pyplot as plt

# Iterate through the unique traits in the mean_results DataFrame
for trait in mean_results2['Trait'].unique():
    # Create a DataFrame with data for the current trait
    trait_data = mean_results2[mean_results2['Trait'] == trait]
    
    # Sort the data by RMSE-to-variance ratio in ascending order (lower values first)
    trait_data = trait_data.sort_values('RMSE_to_Variance_Ratio')
    
    # Create a line plot for the current trait
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Window', y='RMSE_to_Variance_Ratio', hue='Stage Combination', data=trait_data)
    
    # Set the title and labels
    plt.title(f'RMSE to Variance Ratio for {trait} by Window')
    plt.xlabel('Window')
    plt.ylabel('RMSE to Variance Ratio')
    
    # Rotate the x-axis labels to prevent overlapping
    plt.xticks(rotation=45, ha='right')
    
    # Show the plot
    plt.show()

# Dictionary to store matrix_difference for each trait
matrix_differences = {}

# Iterate through the unique traits
for trait in traits:
    # Select columns corresponding to the current trait for both pivoted_rmse and pivoted_variance
    trait_rmse_columns = [(trait, stage_comb) for stage_comb in stage_combinations]
    trait_variance_columns = [(trait, stage_comb) for stage_comb in stage_combinations]

    # Calculate the correlation matrices for the selected columns
    trait_rmse_corr = pivoted_rmse[trait_rmse_columns].corr()
    trait_variance_corr = pivoted_variance[trait_variance_columns].corr()

    # Calculate the matrix_difference for the current trait
    trait_matrix_difference = trait_rmse_corr - trait_variance_corr

    # Store the matrix_difference in the dictionary
    matrix_differences[trait] = trait_matrix_difference



def plot_heatmap(matrix_difference, trait):
    # Set up the plot
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(matrix_difference, dtype=bool))

    # Create a heatmap for the matrix_difference with the mask applied
    ax = sns.heatmap(matrix_difference, annot=True, cmap="coolwarm", square=True, cbar_kws={'label': 'Difference'}, mask=mask)

    # Set the plot title and labels
    ax.set_title(f'Matrix Difference Heatmap for {trait}')
    ax.set_xlabel('RMSE Stage Combinations')
    ax.set_ylabel('Variance Stage Combinations')

    # Save the plot as an image file
    plt.show()

# Iterate through the matrix_differences dictionary and create a heatmap for each trait
for trait, matrix_difference in matrix_differences.items():
    plot_heatmap(matrix_difference, trait)


# Iterate through the unique traits in the rmse_df_filtered DataFrame
for trait in rmse_df_filtered['Trait'].unique():
    # Create a DataFrame with data for the current trait
    trait_rmse_data = rmse_df_filtered[rmse_df_filtered['Trait'] == trait]
    trait_variance_data = variance_df_filtered[variance_df_filtered['Trait'] == trait]
    
    # Create a line plot for the current trait (RMSE)
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Window', y='RMSE', hue='Stage Combination', data=trait_rmse_data)
    
    # Set the title and labels
    plt.title(f'RMSE for {trait} by Window')
    plt.xlabel('Window')
    plt.ylabel('RMSE')
    
    # Rotate the x-axis labels to prevent overlapping
    plt.xticks(rotation=45, ha='right')
    
    # Show the plot
    plt.show()

    # Create a line plot for the current trait (Variance)
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='Window', y='Variance', hue='Stage Combination', data=trait_variance_data)
    
    # Set the title and labels
    plt.title(f'Variance for {trait} by Window')
    plt.xlabel('Window')
    plt.ylabel('Variance')
    
    # Rotate the x-axis labels to prevent overlapping
    plt.xticks(rotation=45, ha='right')
    
    # Show the plot
    plt.show()

# Create an empty dictionary to store the data frames
trait_data_frames = {}

# Iterate through the unique traits in the mean_results DataFrame
for trait in mean_results['Trait'].unique():
    # Create a DataFrame with data for the current trait
    trait_data = mean_results[mean_results['Trait'] == trait]
    
    # Sort the data by RMSE-to-variance ratio in ascending order (lower values first)
    trait_data = trait_data.sort_values('RMSE_to_Variance_Ratio')
    
    # Add the trait data frame to the dictionary
    trait_data_frames[trait] = trait_data
    
# Print the data frames for each trait
for trait, data_frame in trait_data_frames.items():
    print(f'Trait: {trait}\n{data_frame}\n')


