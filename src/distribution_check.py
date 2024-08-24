import numpy as np
from scipy.stats import norm, shapiro, anderson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm
from matplotlib import pyplot as plt

def graph():
    residuals = np.load("temp_window_not_abs.npy")
    mean_residuals = residuals.flatten()
    
    # Plot a histogram of the mean residuals
    plt.hist(mean_residuals, bins=20, density=True, alpha=0.6, color='g')
    
    # Fit a normal distribution to the data
    mu, std = norm.fit(mean_residuals)
    
    # Plot the PDF of the normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    
    # Add labels and title
    plt.title("Histogram of Mean Residuals with Fitted Normal Distribution")
    plt.xlabel("Mean Residuals")
    plt.ylabel("Density")
    
    # Show the plot
    plt.show()

def test(test_type, dataset, boundary):
    print("Test type:", test_type)
    residuals = np.load(dataset)
    # residuals = residuals[:, 90:]
    total_yes = 0
    total_no = 0
    for i in range(27):
        data = residuals[i]
        # Perform Shapiro-Wilk test
        if test_type == "shapiro":
            stat, p = shapiro(data)
            if p > boundary:
                total_yes += 1
                print(i, end=", ")
            else:
                total_no += 1
        elif test_type == "anderson":
            result = anderson(data)
            if result.statistic < result.critical_values[3]:
                total_yes += 1
            else:
                total_no += 1
        elif test_type == "bp":
            # Perform Breusch-Pagan test for heteroscedasticity
            X = np.arange(len(data)).reshape(-1, 1)  # Use index as an independent variable for example
            X = sm.add_constant(X)  # Add constant to the model
            model = sm.OLS(data, X).fit()
            bp_test = het_breuschpagan(model.resid, model.model.exog)
            p_value = bp_test[1]  # Extract p-value from the Breusch-Pagan test results
            
            if p_value > boundary:  # Check if p-value is above a threshold (no evidence of heteroscedasticity)
                total_yes += 1
            else:
                print(i, end=", ")
                total_no += 1
        elif test_type == "white":
            # Perform White test for heteroscedasticity
            X = np.arange(len(data)).reshape(-1, 1)  # Use index as an independent variable for example
            X = sm.add_constant(X)  # Add constant to the model
            model = sm.OLS(data, X).fit()
            white_test = het_white(model.resid, model.model.exog)
            p_value = white_test[1]  # Extract p-value from the White test results
            
            if p_value > boundary:  # Check if p-value is above a threshold (no evidence of heteroscedasticity)
                total_yes += 1
                print(p_value, end=", ")
            else:
                print(i, end=", ")
                total_no += 1
    
    print(f"\nYes: {total_yes}, No: {total_no}")

def main():
    # test(test_type="shapiro", dataset="fit_positive.npy", boundary=0.01)
    graph()

if __name__ == "__main__":
    main()