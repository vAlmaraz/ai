import numpy as np
import os
import sys
import csv

PRICE_COLUMN_INDEX = 6
FILE_TRAINING = "../datasets/buildings.csv"
FILE_COEF = "coef.txt"

def main():
    np.set_printoptions(precision=1)

    if not os.path.isfile(FILE_TRAINING):
        print("Missing training file")
        sys.exit()

    with open(FILE_TRAINING) as file:
        reader = csv.reader(file)
        next(reader)
        data = []
        for row in reader:
            try:
                values = [float(val) for val in row]
                data.append(values)
            except ValueError:
                pass
        data = np.array(data)

    prices = data[:,PRICE_COLUMN_INDEX]
    values = np.delete(data, PRICE_COLUMN_INDEX, 1)

    outliers_indexes = get_outliers_indexes(prices)

    for index in sorted(outliers_indexes, reverse=True):
        del prices[index]
        del values[index]

    c = np.linalg.lstsq(values, prices, rcond=None)[0]

    np.savetxt(FILE_COEF, c)

def get_outliers_indexes(prices, threshold=3):
    mean = np.mean(prices)
    std = np.std(prices)
    
    min_limit = mean - (std * threshold)
    max_limit = mean + (std * threshold)

    return [idx for idx, price in enumerate(prices) if price < min_limit or price > max_limit]

main()