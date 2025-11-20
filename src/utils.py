import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Training History")
    plt.show()
