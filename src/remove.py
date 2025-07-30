input_path = r'c:\Users\sadboiz21\Desktop\500-new.json'
output_path = r'c:\Users\sadboiz21\Desktop\500-new-no-id.json'

with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if '"id":' not in line:
            f_out.write(line)