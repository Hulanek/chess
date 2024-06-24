input_file = 'chess_data_extracted.txt'
output_file = 'chess_data_extracted_deviated.txt'
with open(input_file, 'r') as l:
    aloha = l.readlines()
    with open(output_file, 'w') as f:
        for i in range(len(aloha)):
            if i % 2 == 1:
                aloha[i] = float(aloha[i])
                f.write(f'{aloha[i]/16000}\n')
            else:
                f.write(f'{aloha[i]}')





