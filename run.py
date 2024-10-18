import os
import subprocess

def run_python_files(directory):
    python_files = [f for f in os.listdir(directory) if f.endswith('.py')]
    for file in python_files:
        subprocess.run(['python', os.path.join(directory, file)])

data_fetching_folder = 'data_fetching_files'
run_python_files(data_fetching_folder)

new_material_folder = 'new_material'
pdf_script =  'pdf.py'
pdf_files = [f for f in os.listdir(new_material_folder) if f.endswith('.pdf')]
for pdf_file in pdf_files:
    subprocess.run(['python', pdf_script, os.path.join(new_material_folder, pdf_file)])

new_material_text_folder = 'new_material_text'
similarity_script = 'similarity.py'
text_files = [f for f in os.listdir(new_material_text_folder) if f.endswith('.txt')]
for text_file in text_files:
    subprocess.run(['python', similarity_script, os.path.join(new_material_text_folder, text_file)])


