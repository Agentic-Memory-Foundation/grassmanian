import os
import zipfile

def build_arxiv_zip():
    print("Building arXiv submission zip for camman...")
    
    zip_filename = "arxiv.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    print(f"Adding files to {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add main source files
        files_to_add = [
            "paper.tex",
            "paper.bbl",
            "references.bib",
            "acmart.cls",
            "ACM-Reference-Format.bst"
        ]
        
        for filename in files_to_add:
            if os.path.exists(filename):
                zipf.write(filename)
                print(f"Added {filename}")
            else:
                print(f"Warning: {filename} not found.")
                
        # Add generated tables
        generated_dir = "generated"
        if os.path.exists(generated_dir):
            for root, dirs, files in os.walk(generated_dir):
                for file in files:
                    if file.endswith(".tex"):
                        filepath = os.path.join(root, file)
                        zipf.write(filepath)
                        print(f"Added {filepath}")
        else:
            print("Warning: generated/ directory not found.")
            
        # Add figures
        figures_dir = "figures"
        if os.path.exists(figures_dir):
            for root, dirs, files in os.walk(figures_dir):
                for file in files:
                    if file.endswith(".pdf"):
                        filepath = os.path.join(root, file)
                        zipf.write(filepath)
                        print(f"Added {filepath}")
        else:
            print("Warning: figures/ directory not found.")
            
    print(f"Successfully generated {zip_filename} in camman! (Size: {os.path.getsize(zip_filename)/1024:.1f} KB)")

if __name__ == '__main__':
    build_arxiv_zip()
