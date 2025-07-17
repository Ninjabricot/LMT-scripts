import csv
import os

input_path = r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Sasha\Social LMT\Social_Replacement_Behaviour\Pre_SR\PSR11_12_LMT3_M1_20250416\Events_M1\4_22_8_32_50.csv"
output_dir = r"C:\Users\I9_1\Desktop\LMT"
input_filename = os.path.basename(input_path)
name, ext = os.path.splitext(input_filename)
output_filename = f"{name}_modified{ext}"
output_path = os.path.join(output_dir, output_filename)

os.makedirs(output_dir, exist_ok=True)

with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
     open(output_path, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile, delimiter=';')

    for row in reader:
        if len(row) >= 3:
            # Ajout de :000 à la fin du champ date si ce n'est pas déjà présent
            if not row[2].endswith(':000'):
                row[2] = row[2] + ':000'
        writer.writerow(row)

print(f"Fichier modifié sauvegardé ici : {output_path}")
