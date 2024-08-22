import pandas as pd
import re
import tkinter as tk
from tkinter import filedialog, messagebox


def load_premetadata():
    # Identifier columns
    required_columns = ["batch", "arena", "animal_in_event_record", "Video Name"]

    # Open file dialog to select metadata file
    file_path = filedialog.askopenfilename(title="Select pre-existing metadata file")

    if not file_path:
        tk.messagebox.showwarning("Warning", "No file selected.")
        return

    print("Loading pre-existing metadata from:", file_path)
    pre_mt = pd.read_excel(file_path, dtype={i: str for i in required_columns})

    # Check that the required columns are in there
    if not all(c in pre_mt.columns for c in required_columns):
        tk.messagebox.showwarning(
            "Warning",
            f"Pre-existing metadata excel does not have the columns {required_columns}. Include them before continuing.",
        )
        return

    # Create a pop-up window for column selection
    column_selection_window = tk.Tk()
    column_selection_window.title("Select Columns")
    print("Make selection of all the metadata columns you want to tranfer to your data")

    selected_columns = []

    def ok_callback():
        nonlocal selected_columns
        selected_columns = [
            pre_mt.columns[i] for i, val in enumerate(checkboxes) if val.get()
        ]
        column_selection_window.destroy()

    checkboxes = []
    for col in pre_mt.columns:
        var = tk.BooleanVar()
        checkboxes.append(var)
        checkbox = tk.Checkbutton(
            column_selection_window,
            text=col,
            variable=var,
            onvalue=True,
            offvalue=False,
        )
        checkbox.pack(anchor=tk.W)

    ok_button = tk.Button(column_selection_window, text="OK", command=ok_callback)
    ok_button.pack()

    column_selection_window.mainloop()
    print(f"Selected columns {selected_columns}")
    if not selected_columns:
        tk.messagebox.showwarning("Warning", "No columns selected.")
        return

    # Slice data based on selection
    pre_mt = pre_mt[list(set(selected_columns + required_columns))]

    # Load main data
    print("Select your main data in order to append the metadata")
    data_path = filedialog.askopenfilename(title="Select pre-processed data file")

    if not data_path:
        tk.messagebox.showwarning("Warning", "No file selected.")
        return
    print(f"Loading data from {data_path}")
    df = pd.read_excel(data_path, dtype={i: str for i in required_columns})

    # Merge and save
    out_name = re.split("\.", data_path)[0]
    out_name = f"{out_name}_with-metadata.xlsx"

    print("Merging and saving")
    pd.merge(df, pre_mt, on=required_columns, how="left").to_excel(
        out_name, index=False
    )


def load_auto_metadata():
    # Identifier columns
    required_columns = ["batch", "arena", "animal_in_event_record", "Video Name"]

    # Function to load auto-generated metadata
    print("Loading auto-generated metadata...")
    file_path = filedialog.askopenfilename(title="Select EDITED auto-metadata file")

    if not file_path:
        tk.messagebox.showwarning("Warning", "No file selected.")
        return

    mt = pd.read_excel(file_path, dtype={i: str for i in required_columns})

    # Load main data
    print("Select your main data in order to append the metadata")
    data_path = filedialog.askopenfilename(title="Select pre-processed data file")

    if not data_path:
        tk.messagebox.showwarning("Warning", "No file selected.")
        return

    print(f"Loading data from {data_path}")
    df = pd.read_excel(data_path, dtype={i: str for i in required_columns})

    # Merge and save
    out_name = re.split("\.", data_path)[0]
    out_name = f"{out_name}_with-metadata.xlsx"

    print("Merging and saving")
    pd.merge(df, mt, on=required_columns, how="left").to_excel(out_name, index=False)


def open_dialog():
    dialog = tk.Tk()
    dialog.title("Select metadata file")
    dialog.geometry("400x100")  # Set the size of the window

    def pre_metadata_callback():
        dialog.destroy()
        load_premetadata()

    def auto_metadata_callback():
        dialog.destroy()
        load_auto_metadata()

    pre_metadata_button = tk.Button(
        dialog, text="I have pre-existing metadata", command=pre_metadata_callback
    )
    pre_metadata_button.pack(pady=10)

    auto_metadata_button = tk.Button(
        dialog,
        text="I've already filled in the auto-generated metadata excel",
        command=auto_metadata_callback,
    )
    auto_metadata_button.pack(pady=10)

    dialog.mainloop()


if __name__ == "__main__":

    open_dialog()
    print("Done")
