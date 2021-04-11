import matplotlib.pyplot as plt
import sys


def main(argv):
    plt.ion()
    fig = plt.figure(figsize=(20, 10))
    plt.show()

    for _ in range(2):
        random_data = [123, 321, 3213, 131]
        plt.plot(random_data)
        plt.draw()
        plt.pause(0.001)
        print("something after plotting")
        _ = input("perssing any button to continue")


if __name__ == "__main__":
    main(sys.argv[1:])
