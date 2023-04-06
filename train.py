import numpy as np
import os
import sys
import csv

PRICE_COLUMN_INDEX = 6
FILE_TRAINING = "Prediccion_pisos.csv"
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
                values = [float(val) if idx != len(row)-1 else val for idx, val in enumerate(row[0:-1])]
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

    outliers_indexes = []
    for idx, price in enumerate(prices):
        if price < min_limit or price > max_limit:
            outliers_indexes.append(idx)
    
    return outliers_indexes

    while True:
        try:
            m2 = float(input(f"Cuántos metros cuadrados tiene el piso ({RANGE_M2[0]}-{RANGE_M2[1]})? "))
            if m2 < RANGE_M2[0] or m2 > RANGE_M2[1]:
                raise ValueError()
            break
        except ValueError:
            print(f"Error: los metros cuadrados deben estar entre {RANGE_M2[0]} y {RANGE_M2[1]}")

    while True:
        try:
            rooms = int(input(f"Cuántas habitaciones tiene ({RANGE_ROOMS[0]}-{RANGE_ROOMS[1]})? "))
            if rooms < RANGE_ROOMS[0] or rooms > RANGE_ROOMS[1]:
                raise ValueError()
            break
        except ValueError:
            print(f"Error: el número de habitaciones debe estar entre {RANGE_ROOMS[0]} y {RANGE_ROOMS[1]}")

    while True:
        try:
            floor = int(input(f"Cuál es la planta del piso ({RANGE_FLOOR[0]}-{RANGE_FLOOR[1]})? "))
            if floor < RANGE_FLOOR[0] or floor > RANGE_FLOOR[1]:
                raise ValueError()
            break
        except ValueError:
            print(f"Error: la planta del piso debe estar entre {RANGE_FLOOR[0]} y {RANGE_FLOOR[1]}")

    while True:
        try:
            elevator = int(input(f"Tiene ascensor? ({RANGE_ELEVATOR[0]} no -{RANGE_ELEVATOR[1]} sí): "))
            if elevator < RANGE_ELEVATOR[0] or elevator > RANGE_ELEVATOR[1]:
                raise ValueError()
            break
        except ValueError:
            print(f"Error: el valor del ascensor debe estar entre {RANGE_ELEVATOR[0]} y {RANGE_ELEVATOR[1]}")

    return np.array([37, 1, m2, rooms, floor, elevator])

main()