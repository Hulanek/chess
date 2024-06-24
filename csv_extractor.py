
import csv

with open('chessData.csv') as csvfile:
    reader = csv.reader(csvfile)
    with open('chess_data_extracted.txt', 'w') as exfile:
        tempval_positive = 0
        tempval_negative = 0
        tempval_nohash = 0
        tempval_nohash_negative = 0
        for lines in csvfile:
            a = lines.split(',')
            if a[0] == 'FEN':
                continue
            exfile.write(a[0] + '\n')
            if '#' in a[1]:
                b = a[1]
                if '+' in b:
                    c = 8000
                    exfile.write(f'{c}\n')
                elif '-' in b:
                    c = -8000
                    exfile.write(f'{c}\n')
            else:
                d = float(a[1])
                if d > tempval_nohash:
                    tempval_nohash = d
                elif d < tempval_nohash_negative:
                    tempval_nohash_negative = d
                exfile.write(a[1])
    print(f'highest score: {tempval_positive}, lowest score: {tempval_negative}')
    print(f'nohash_positive: {tempval_nohash}, nohasnegative {tempval_nohash_negative}')

