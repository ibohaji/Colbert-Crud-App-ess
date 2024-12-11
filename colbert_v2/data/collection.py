# data/collection.py
class FileProcessor:
    def __init__(self, path):
        self.path = path

    def load_collection(self):
        documents = []
        with open(self.path) as f:
            for line in f:
                columns = line.strip().split('\t')
                documents.append(columns)
        return documents
    @staticmethod
    def ensure_proper_header(input_file):
        with open(input_file) as f_in:
            lines = f_in.readlines()

        # Ensure the first line is exactly 'id\ttext' and the rest remains unchanged
        if lines[0].strip() != 'id\ttext':
            lines.insert(0, "id\ttext\n")

        # Write the updated content back to the same file
        with open(input_file, 'w') as f_out:
            f_out.writelines(lines)

    @staticmethod
    def add_missing_headers(input_file):
        with open(input_file) as f_in:
            lines = f_in.readlines()

        with open(input_file, 'w') as f_out:
            for line_idx, line in enumerate(lines):
                # If it's the first line and it's already the header, leave it as is
                if line_idx == 0 and line.strip() == "id\ttext":
                    f_out.write(line)  # Write the existing header
                else:
                    # For all other lines, split by tab, replace the first value with the line index
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        f_out.write(f"{line_idx}\t{parts[1]}\n")



    def save_new_document(self, new_document):
        with open(self.path, 'a') as f:
            f.write(f"\n{new_document}")
