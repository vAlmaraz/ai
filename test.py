import numpy as np
import os


FILE_COEF = "coef.txt"
RANGE_M2 = (10, 800)
RANGE_ROOMS = (1, 10)
RANGE_FLOOR = (0, 15)
RANGE_ELEVATOR = (0, 1)

def main():
    np.set_printoptions(precision=1)

    if not os.path.isfile(FILE_COEF):
        print("Missing coef file. Training...")
        os.system("python train.py")

    c = np.loadtxt(FILE_COEF)

    building = ask_building_data()

    print("El precio predecido para ese piso es de", round(building @ c, 2), "€")

def ask_building_data():

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