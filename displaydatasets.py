import pickle
import tkinter as tk
from tkinter import scrolledtext, messagebox

# File path (fixed)
file_path = 'datasets/datasets.pkl'

# Load the pickle
with open(file_path, 'rb') as f:
    sentences = pickle.load(f)

# Remove duplicates automatically and save back to file
initial_count = len(sentences)
sentences = list(dict.fromkeys(sentences))  # Remove duplicates, keep order
removed_count = initial_count - len(sentences)

if removed_count > 0:
    with open(file_path, 'wb') as f:
        pickle.dump(sentences, f)
    print(f"🗑️ Removed {removed_count} duplicate entries and saved to file.")
else:
    print("✅ No duplicates found.")

# Create window
root = tk.Tk()
root.title("Pickle Data Viewer")

# Add Scrolled Text widget (Editable by default)
text_area = scrolledtext.ScrolledText(root, width=160, height=40)
text_area.pack(padx=10, pady=(10, 0))

# Insert sentences
for sentence in sentences:
    text_area.insert(tk.END, sentence + '\n')

# Label to display total lines
line_count_label = tk.Label(root, text="")
line_count_label.pack(pady=(5, 10))

# Function to update line count
def update_line_count():
    num_lines = int(text_area.index('end-1c').split('.')[0])
    line_count_label.config(text=f"Total Lines: {num_lines}")

# Save function (overwrite file)
def save_changes():
    edited_text = text_area.get("1.0", tk.END).strip().split('\n')
    with open(file_path, 'wb') as f:
        pickle.dump(edited_text, f)
    update_line_count()
    messagebox.showinfo("Success", f"Changes saved to {file_path}")

# Save button
save_button = tk.Button(root, text="Save Changes", command=save_changes)
save_button.pack(pady=(0, 10))

# Update line count initially and after key edits
update_line_count()
text_area.bind('<KeyRelease>', lambda e: update_line_count())

# Run the window
root.mainloop()
