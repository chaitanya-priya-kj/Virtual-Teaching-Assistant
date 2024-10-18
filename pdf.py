import os
import sys
from PyPDF2 import PdfReader

def convert_pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pdf.py <pdf_files_folder>")
        sys.exit(1)

    pdf_files_folder = sys.argv[1]

    if pdf_files_folder == 'pdf_files':
    # Create study_material folder if it doesn't exist
        folder_path = 'study_material'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # List of PDF files in the specified folder
        pdf_files = [file for file in os.listdir(pdf_files_folder) if file.endswith('.pdf')]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_files_folder, pdf_file)
            text = convert_pdf_to_text(pdf_path)

            # Save text to a text file in study_material folder with the same name
            text_file_path = os.path.join(folder_path, os.path.splitext(pdf_file)[0] + '.txt')
            save_text_to_file(text, text_file_path)

        print("PDF files converted to text and saved in study_material folder.")
    else:
            # Create study_material folder if it doesn't exist
        pdf_files_folder = 'new_material'
        folder_path = 'new_material_text'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # List of PDF files in the specified folder
        pdf_files = [file for file in os.listdir(pdf_files_folder) if file.endswith('.pdf')]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_files_folder, pdf_file)
            text = convert_pdf_to_text(pdf_path)

            # Save text to a text file in study_material folder with the same name
            text_file_path = os.path.join(folder_path, os.path.splitext(pdf_file)[0] + '.txt')
            save_text_to_file(text, text_file_path)

        print("PDF files converted to text and saved in new_material_text folder.")




# import os
# from PyPDF2 import PdfReader

# def convert_pdf_to_text(pdf_path):
#     text = ""
#     with open(pdf_path, 'rb') as file:
#         reader = PdfReader(file)
#         num_pages = len(reader.pages)
#         for page_num in range(num_pages):
#             page = reader.pages[page_num]
#             text += page.extract_text()
#     return text

# def save_text_to_file(text, filename):
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write(text)

# if __name__ == "__main__":
#     # Create study_material folder if it doesn't exist
#     folder_path = 'study_material'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     # Folder containing PDF files
#     pdf_files_folder = 'pdf_files'

#     # List of PDF files in the specified folder
#     pdf_files = [file for file in os.listdir(pdf_files_folder) if file.endswith('.pdf')]

#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_files_folder, pdf_file)
#         text = convert_pdf_to_text(pdf_path)

#         # Save text to a text file in study_material folder with the same name
#         text_file_path = os.path.join(folder_path, os.path.splitext(pdf_file)[0] + '.txt')
#         save_text_to_file(text, text_file_path)

#     print("PDF files converted to text and saved in study_material folder.")
