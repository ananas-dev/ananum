import os

def remove_identifier_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".Identifier"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    remove_identifier_files(current_directory)