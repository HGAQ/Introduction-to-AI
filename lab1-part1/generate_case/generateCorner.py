import sys
import random
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 3:
        print("He who rides a horse must provide its feed.")
    else:
        right = int(args[0])
        top = int(args[1])
        possibility = float(args[2])
        px, py = random.randint(4, right-3), random.randint(4, top-3)
        not_walls = []
        for i in range(right+2):
            not_walls.append([0]*(top+2))
        not_walls[px][py] = not_walls[1][1] = not_walls[1][top] = not_walls[right][1] = not_walls[right][top] = 1
        threshold = random.randint(right, right + top)
        def generateRoute(x, y, edx, edy, direction):
            for i in range(threshold):
                x0 = x
                y0 = y
                while True:
                    select = random.randint(0, len(direction)-1)
                    x = x0 + direction[select][0]
                    y = y0 + direction[select][1]
                    if 1 <= x <= right and 1 <= y <= top:
                        break
                not_walls[x][y] = 1
                if x == edx and y == edy:
                    return
            for i in range(min(x, edx), max(x, edx)+1):
                for j in range(min(y, edy), max(y, edy)+1):
                    not_walls[i][j] = 1

        generateRoute(px, py, 1, 1, [(-1, 0), (1, 0), (0, -1)])
        generateRoute(1, 1, 1, top, [(0, 1), (1, 0), (-1, 0)])
        generateRoute(1, top, right, top, [(0, 1), (1, 0), (0, -1)])
        generateRoute(right, top, right, 1, [(0, -1), (1, 0), (-1, 0)])
        with open("mediumCorners.lay", "w") as fout:
            fout.write("%"*(right+2)+"\n")
            for i in range(top, 0, -1):
                fout.write("%")
                for j in range(1, right+1):
                    if not not_walls[j][i]:
                        if random.random() < possibility:
                            fout.write("%")
                        else:
                            fout.write(" ")
                    elif j == px and i == py:
                        fout.write("P")
                    elif (i == 1 or i == top) and (j == 1 or j == right):
                        fout.write(".")
                    else:
                        fout.write(" ")
                fout.write("%\n")
            fout.write("%"*(right+2))