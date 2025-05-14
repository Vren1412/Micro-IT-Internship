import tkinter as tk
from tkinter import messagebox
# Main window setup
root = tk.Tk()
root.title("Simple Calculator")
root.configure(bg="#fdf6f0")  # Soft peach background
root.resizable(False, False)
expression = ""
# Entry display field (moderate size)
entry = tk.Entry(
    root, width=22, font=("Calibri", 24),
    borderwidth=6, relief="flat",
    justify='right', bg="#fffaf7", fg="#333"
)
entry.grid(row=0, column=0, columnspan=4, padx=12, pady=12, ipady=12)
# Functions
def button_click(symbol):
    global expression
    expression += str(symbol)
    entry.delete(0, tk.END)
    entry.insert(0, expression)
def calculate():
    global expression
    try:
        result = str(eval(expression))
        entry.delete(0, tk.END)
        entry.insert(0, result)
        expression = result
    except ZeroDivisionError:
        messagebox.showerror("Error", "Division by zero is undefined.")
        clear()
    except:
        messagebox.showerror("Error", "Invalid input.")
        clear()
def clear():
    global expression
    expression = ""
    entry.delete(0, tk.END)
# Button layout
buttons = [
    ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
    ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
    ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
    ('0', 4, 0), ('.', 4, 1), ('C', 4, 2), ('+', 4, 3),
    ('=', 5, 0, 4)
]
# Color palette
color_palette = {
    'numbers': "#e8f6ef",     
    'operators': "#dee7f2",   
    'equal': "#f6d6d6",       
    'clear': "#fde2e4",       
    'fg_dark': "#2d3436",     
    'fg_light': "#ffffff"     
}
# Rounded button factory
def create_rounded_button(master, text, command, bg, fg):
    return tk.Button(
        master,
        text=text,
        font=("Calibri", 18, "bold"),
        width=5,
        height=2,
        bg=bg,
        fg=fg,
        activebackground="#dfe6e9",
        bd=0,
        relief="flat",
        highlightthickness=0,
        padx=10,
        pady=10,
        command=command
    )
# Create calculator buttons
for btn in buttons:
    text, row, col = btn[0], btn[1], btn[2]
    colspan = btn[3] if len(btn) == 4 else 1
    # Determine colors
    if text == '=':
        bg = color_palette['equal']
        fg = color_palette['fg_dark']
    elif text == 'C':
        bg = color_palette['clear']
        fg = color_palette['fg_dark']
    elif text in ('+', '-', '*', '/'):
        bg = color_palette['operators']
        fg = color_palette['fg_dark']
    else:
        bg = color_palette['numbers']
        fg = color_palette['fg_dark']
    # Function mapping
    action = calculate if text == '=' else clear if text == 'C' else lambda x=text: button_click(x)
    # Create rounded button
    btn_widget = create_rounded_button(root, text, action, bg, fg)
    btn_widget.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=6, pady=6)
# Grid configuration
for i in range(6):
    root.grid_rowconfigure(i, weight=1)
for j in range(4):
    root.grid_columnconfigure(j, weight=1)
# Run app
root.mainloop()
