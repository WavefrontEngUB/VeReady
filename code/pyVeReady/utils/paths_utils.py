import tkinter
import tkinter.filedialog
import os
import re
from datetime import datetime


def ask_yes_no_prompt(message, default_answer):
    """
    Display a yes/no dialog asking the user to confirm an action.

    Parameters:
        message (str): The message to display in the dialog.
        default_answer (str, optional): The default response ('yes' or 'no').

    Returns:
        bool: True if the user selects 'Yes', False if 'No'.
    """
    root = tkinter.Tk()
    root.withdraw()  # Hide the main Tkinter window

    return tkinter.messagebox.askyesno('Prompt', message, default=default_answer)

def ask_files_location(message, initial_directory=None, return_first_string=False):
    """
    Prompt the user to select one or more file locations through a dialog box.

    Parameters:
        message (str): The title message displayed on the file dialog.
        initial_directory (str, optional): The initial directory to select. If not provided, initial directory is the current working directory.
        return_first_string (bool, optional): If True, the first string path will be returned.

    Returns:
        list: A list of selected file paths as strings. If a single file is selected, the list contains one item.
    """
    root = tkinter.Tk()
    file_locations = tkinter.filedialog.askopenfilenames(parent=root, title=message,
                                                         initialdir=initial_directory if initial_directory is not None else os.getcwd())
    root.withdraw()
    root.destroy()
    if not file_locations or file_locations[0] == "":
        return None
    elif return_first_string:
        return file_locations[0]
    else:
        return list(file_locations)

def ask_directory_location(message):
    """
    Prompt the user to select a single directory location through a dialog box.

    Parameters:
        message (str): The title message displayed on the directory dialog.

    Returns:
        str: The path of the selected directory.
    """
    root = tkinter.Tk()
    file_location = tkinter.filedialog.askdirectory(parent=root, title=message)
    root.withdraw()
    return file_location

def get_unique_filename(filename, directory=None):
    """Ensures a unique filename by appending a number before the extension if needed."""
    if directory is None:
        directory = os.getcwd()  # Default to the current working directory

    base, ext = os.path.splitext(filename)  # Preserve full base name
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_({counter}){ext}"
        counter += 1

    return new_filename

def add_date_prefix(filename):
    """Prepends today's date (YYYYMMDD) to the given filename."""
    today_date = datetime.today().strftime('%Y%m%d')
    return f"{today_date}_{filename}"