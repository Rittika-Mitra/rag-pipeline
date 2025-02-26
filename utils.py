import os

def process_path(path):
    # Check if the path exists
    if not os.path.exists(path):
        print(f"The path '{path}' does not exist.")
        return
    
    # Check if the path is a file
    if os.path.isfile(path):
        # Check if the file is a PDF
        if path.lower().endswith('.pdf'):
            print(f"'{path}' ---- Processing...")
            return True
        else:
            print(f"'{path}' is a not a PDF file.")
    elif os.path.isdir(path):
        print(f"'{path}' is a directory. Iterating over files...")
        return "dir"
    else:
        print(f"'{path}' is neither a file nor a directory.")

if __name__ == '__main__':
    path = input("Enter the file or directory path: ").strip()
    process_path(path)
