import matplotlib.pyplot as plt
import pandas as pd

# Read the uploaded Excel file
data = pd.read_excel("2.xlsx")
data.head()

# Check if FedPPN is in the data
if "FedPPN" in data["Unnamed: 0"].values:
    # Separate FedPPN from other methods
    fedppn_data = data[data["Unnamed: 0"] == "FedPPN"]
    other_data = data[data["Unnamed: 0"] != "FedPPN"]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(other_data["Client total exchanged bytes"], other_data["Accuracy"], label="Other Methods", s=100)
    plt.scatter(fedppn_data["Client total exchanged bytes"], fedppn_data["Accuracy"], color="red", marker="*", label="FedPPN",
                s=200)

    # Annotate methods
    for _, row in data.iterrows():
        plt.annotate(row["Unnamed: 0"], (row["Client total exchanged bytes"], row["Accuracy"]), fontsize=9, ha='right')

    plt.xlabel("Per client communication overhead (MB)",fontdict={'family' : 'Times New Roman', 'size':15})
    plt.ylabel("Accuracy",fontdict={'family' : 'Times New Roman', 'size':15})
    plt.title("Communications cost vs. Accuracy",fontdict={'family' : 'Times New Roman', 'size':20})
    # plt.legend()
    plt.grid(True)
    plt.show()
else:
    # If FedPPN is not in the data
    plt.figure(figsize=(10, 6))
    plt.scatter(data["Client total exchanged bytes"], data["Accuracy"], label="Methods", s=100)
    for _, row in data.iterrows():
        plt.annotate(row["Unnamed: 0"], (row["Client total exchanged bytes"], row["Accuracy"]), fontsize=9, ha='right')
    plt.xlabel("Client total exchanged bytes")
    plt.ylabel("Accuracy")
    plt.title("Communications cost vs. Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
