import GaraCossEuler as gc


bb = gc.GaraCossEuler(62,3.4)

while True:
    try:
        tdl = float(input('Enter TDL: '))
        x, y = bb.getTipPos(tdl)
        print(f'X: {x:.3f}, Y: {y:.3f}')
    except ValueError:
        print('Invalid input. Please try again.')
    else:
        break