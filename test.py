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

    m2 = ask_integer(
        f"Cuántos metros cuadrados tiene el piso ({RANGE_M2[0]}-{RANGE_M2[1]})? ",
        RANGE_M2,
        f"Error: los metros cuadrados deben estar entre {RANGE_M2[0]} y {RANGE_M2[1]}"
    )

    rooms = ask_integer(
        f"Cuántas habitaciones tiene ({RANGE_ROOMS[0]}-{RANGE_ROOMS[1]})? ",
        RANGE_ROOMS,
        f"Error: el número de habitaciones debe estar entre {RANGE_ROOMS[0]} y {RANGE_ROOMS[1]}"
    )

    floor = ask_integer(
        f"Cuál es la planta del piso ({RANGE_FLOOR[0]}-{RANGE_FLOOR[1]})? ",
        RANGE_FLOOR,
        "Error: la planta del piso debe estar entre {RANGE_FLOOR[0]} y {RANGE_FLOOR[1]}"
    )

    elevator = ask_integer(
        f"Tiene ascensor ({RANGE_ELEVATOR[0]} si es que no, {RANGE_ELEVATOR[1]} para sí)? ",
        RANGE_ELEVATOR,
        f"Error: el valor del ascensor debe estar entre {RANGE_ELEVATOR[0]} y {RANGE_ELEVATOR[1]}"
    )

    return np.array([37, 1, m2, rooms, floor, elevator])

def ask_integer(question, range, error_message):
    while True:
        try:
            the_integer = int(input(question))
            if the_integer < range[0] or the_integer > range[1]:
                raise ValueError()
            break
        except ValueError:
            print(error_message)
    return the_integer

main()