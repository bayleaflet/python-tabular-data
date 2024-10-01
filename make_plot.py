import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def subset_by_species(dataframe, species_name):
    '''Subsets data set by species'''
    return dataframe[dataframe.species == species_name]

def get_regression(subset_df):
    '''Function to get regression of given subset'''
    x = subset_df.petal_length_cm
    y = subset_df.sepal_length_cm
    regression = stats.linregress(x, y)
    return regression

def plot_regression(subset_df, regression, species_name):
    '''Function to plot the data and  regression line'''
    x = subset_df.petal_length_cm
    y = subset_df.sepal_length_cm
    plt.scatter(x, y, label='Data')
    plt.plot(x, regression.slope * x + regression.intercept, color="orange", label='Fitted line')
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(f"Regression for {species_name}")
    plt.legend()
    plt.savefig(f"{species_name}_petal_v_sepal_length_regress.png")
    plt.clf()  # Clearsn plot for next iteration

def main():
    '''Main function that calls other functions to plot data'''
    dataframe = pd.read_csv("iris.csv")
    species_list = ["Iris_versicolor","Iris_virginica", "Iris_setosa"]
    for species in species_list:
        subset_df = subset_by_species(dataframe, species)
        regression = get_regression(subset_df)
        plot_regression(subset_df, regression, species)


if __name__ == '__main__':
    main()
