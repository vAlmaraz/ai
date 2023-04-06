import numpy as np
import csv

PRICE_COLUMN_INDEX = 6
RANGE_M2 = (10, 800)
RANGE_ROOMS = (1, 10)
RANGE_FLOOR = (0, 15)
RANGE_ELEVATOR = (0, 1)

def main():
    np.set_printoptions(precision=1)

    with open("Prediccion_pisos.csv") as file:
        reader = csv.reader(file)
        header = next(reader)
        data = []
        for row in reader:
            try:
                values = [float(val) if idx != len(row)-1 else val for idx, val in enumerate(row[1:-1])]
                data.append(values)
            except ValueError:
                print("a")
                pass  # omitir la fila si no se puede convertir a float
        data = np.array(data)

    prices = data[:,PRICE_COLUMN_INDEX]
    values = np.delete(data, PRICE_COLUMN_INDEX, 1)

    c = np.linalg.lstsq(values, prices, rcond=None)[0]

    # print(values @ c)

    building = askBuildingData()

    print ("El precio predecido para ese piso es de", round(building @ c, 2), "€")

def askBuildingData():

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