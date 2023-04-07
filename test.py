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

    print("The predicted price for this building is", round(building @ c, 2), "€")

def ask_building_data():

    m2 = ask_integer(
        f"How many m2 does it have ({RANGE_M2[0]}-{RANGE_M2[1]})? ",
        RANGE_M2,
        f"Error: m2 must be a number between {RANGE_M2[0]} and {RANGE_M2[1]}"
    )

    rooms = ask_integer(
        f"How many rooms ({RANGE_ROOMS[0]}-{RANGE_ROOMS[1]})? ",
        RANGE_ROOMS,
        f"Error: rooms must be a number between {RANGE_ROOMS[0]} and {RANGE_ROOMS[1]}"
    )

    floor = ask_integer(
        f"What floor it is ({RANGE_FLOOR[0]}-{RANGE_FLOOR[1]})? ",
        RANGE_FLOOR,
        "Error: floor must be a number between {RANGE_FLOOR[0]} and {RANGE_FLOOR[1]}"
    )

    elevator = ask_integer(
        f"Tiene ascensor ({RANGE_ELEVATOR[0]} si es que no, {RANGE_ELEVATOR[1]} para sí)? ",
        RANGE_ELEVATOR,
        f"Error: elevator must be {RANGE_ELEVATOR[0]} or {RANGE_ELEVATOR[1]}"
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