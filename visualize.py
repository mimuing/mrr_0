import matplotlib.pyplot as plt

def plot_loss_accuracy(losses, accuracies):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(losses, 'g-')
    ax2.plot(accuracies, 'b-')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Accuracy', color='b')

    plt.show()

if __name__ == "__main__":
    losses = [0.9, 0.7, 0.5]  # 예시 데이터
    accuracies = [60, 75, 85]  # 예시 데이터
    plot_loss_accuracy(losses, accuracies)
