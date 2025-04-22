import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#data = pd.read_csv(
#    "E:\\VS_Code\\Intro_Machine_Learning\\Project\\SET1_Water_Quality_1_Year.csv"
#)


data = pd.read_csv(
    "D:\\Pycharm\\IML_Project\\Water_Quality_1Year_TwiceDaily_Extended.csv"
)
df = pd.DataFrame(data)


def data_analysis(df):
    # check the missing value
    # print("Missing values before cleaning:")
    # print(df.isnull().sum())

    # fill the missing values with the mean values
    df["pH"] = df["pH"].ffill()
    df["Temperature"] = df["Temperature"].bfill()
    df["Ammonia"] = df["BOD"].ffill()
    df["BOD"] = df["BOD"].ffill()
    df["COD"] = df["COD"].ffill()
    df["DO"] = df["DO"].ffill()
    # visualize the data
    # summary statistics for all parameters.

    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate Q1, Q3, IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[column] = np.where(df[column] < lower_bound, lower_bound,np.where(df[column] > upper_bound, upper_bound, df[column]))
        print(f"Column '{column}': Replaced outliers with bounds [{lower_bound:.2f}, {upper_bound:.2f}]")

        # Cap outliers
        #df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        #print(f"Column '{column}': Outliers capped at [{lower_bound:.2f}, {upper_bound:.2f}]")

    df.to_csv("New_Water_Quality_AfterCleaning.csv", index=False)


def heatmap_analysis(df):
    # heatmap
    t_corr = df.corr(numeric_only=True)
    t_corr.columns = [col[:5] for col in t_corr.columns]
    plt.figure(figsize=(10, 8))
    sb.heatmap(t_corr, annot=True, cmap="Oranges", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


def plot_histogram(df):

    # plot a histogram for each parameter
    columns_to_plot = [col for col in df.columns if col != "Timestamp"]

    # Loop through each column to create separate figures
    for col in columns_to_plot:
        # Create a new figure for each parameter
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot histogram (left subplot)
        axes[0].hist(df[col], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        axes[0].set_title(f"Histogram of {col}", fontsize=12)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # Plot box plot (right subplot)
        sb.boxplot(x=df[col], color="orange", fliersize=5, width=0.3, ax=axes[1])
        axes[1].set_title(f"Box Plot of {col}", fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def time_Series(df):
    time = df["Timestamp"]  # Assuming 'Time' column exists in the dataframe
    ph = df["pH"]
    Amn = df["Ammonia"]
    Temp = df["Temperature"]
    ORP = df["ORP"]
    WL = df["Water_Level"]
    columns_to_plot = [col for col in df.columns if col != "Timestamp"and col != "pH"]

    for col in columns_to_plot:
        # Create a new figure for each parameter
        fig, axes = plt.subplots(1, 1, figsize=(12, 4))

        plt.plot(time, df["pH"], label="pH", marker='o', linestyle='-', color="green")
        plt.plot(time, df[col], label="Ammonia", marker='o', linestyle='-', color="blue")
        plt.title(f"PH value and {col} Over Time", fontsize=12)
        plt.xlabel(col)
        plt.ylabel(f"pH and {col}")
        plt.xticks(rotation=45)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def scattor(df):
    ph = df["pH"]
    Amn = df["Ammonia"]
    Temp = df["Temperature"]
    ORP = df["ORP"]
    WL = df["Water_Level"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot for pH vs Ammonia
    

    sb.regplot(
        ax=axes[0, 0], x=ph, y=Amn,
        scatter_kws={"s": 100, "color": "green", "edgecolor": "black", "alpha": 0.75},
        line_kws={"color": "red", "linewidth": 2, "linestyle": "--"}
    )
    axes[0, 0].set_title("pH vs Ammonia with trend line")
    axes[0, 0].set_xlabel("pH")
    axes[0, 0].set_ylabel("Ammonia")

    # Scatter plot for pH vs Temperature
    sb.regplot(
        ax=axes[0, 1], x=ph, y=Temp,
        scatter_kws={"s": 100, "color": "blue", "edgecolor": "black", "alpha": 0.75},
        line_kws={"color": "red", "linewidth": 2, "linestyle": "--"}
    )
    axes[0, 1].set_title("pH vs Temperature with trend line")
    axes[0, 1].set_xlabel("pH")
    axes[0, 1].set_ylabel("Temperature")

    # Scatter plot for pH vs ORP
    sb.regplot(
        ax=axes[1, 0], x=ph, y=ORP,
        scatter_kws={"s": 100, "color": "red", "edgecolor": "black", "alpha": 0.75},
        line_kws={"color": "blue", "linewidth": 2, "linestyle": "--"}
    )
    axes[1, 0].set_title("pH vs ORP with trend line with trend line")
    axes[1, 0].set_xlabel("pH")
    axes[1, 0].set_ylabel("ORP")

    # Scatter plot for pH vs Water Level
    sb.regplot(
        ax=axes[1, 1], x=ph, y=WL,
        scatter_kws={"s": 100, "color": "purple", "edgecolor": "black", "alpha": 0.75},
        line_kws={"color": "red", "linewidth": 2, "linestyle": "--"}
    )
    axes[1, 1].set_title("pH vs Water Level")
    axes[1, 1].set_xlabel("pH")
    axes[1, 1].set_ylabel("Water Level")

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()



if __name__ == "__main__":
    data_analysis(df)
    print(df)
    ds = df[["pH", "Ammonia", "Temperature", "ORP", "Water_Level"]].describe()
    print(ds)
    scattor(df)
    time_Series(df)
    heatmap_analysis(df)
    plot_histogram(df)

else:
    i = 0
    print("Loading the data...")
    data_analysis(df)
    for index in range(3):
        print("...................")

    print("Loading complete !!!")
