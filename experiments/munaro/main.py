from collector import Collector, CreateTrainTest


base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset/'
file = 'final_la_haute_R0711.csv'


if __name__ == "__main__":
    collector = Collector(base_path)
    df = collector.read_data(file)

    create_train_test = CreateTrainTest()
    # X and y are train samples. X1, y1 are test samples.
    X, y, X1, y1 = create_train_test.create_data(df, months=[1, 2], look_back=8, data_partition=0.8)