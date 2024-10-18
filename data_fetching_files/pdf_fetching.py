import os
import requests
from googlesearch import search

def search_pdf_links(query):
    search_results = search(query, stop=5)
    pdf_links = [link for link in search_results if link.endswith('.pdf')]
    return pdf_links

def download_pdfs(links, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for idx, link in enumerate(links, 1):
        try:
            response = requests.get(link)
            file_name = os.path.join(folder_path, f'data_science_material_{idx}.pdf')
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded: {file_name}')
        except Exception as e:
            print(f'Error downloading {link}: {e}')

def main():
    query = 'Data Science study material machine learning filetype:pdf'
    folder_path = 'new_material' 
    
    pdf_links = search_pdf_links(query)

    download_pdfs(pdf_links, folder_path)

if __name__ == "__main__":
    main()
