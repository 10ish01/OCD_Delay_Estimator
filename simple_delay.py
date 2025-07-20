# Run this in-case you want to use a simple delay estimator that doesnt rely on any past data and gives a general blanket maximum delay estimation
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

def predict_delay():
    try:
        samples = int(entry_samples.get())
        avg_tests = float(entry_avg_tests.get())
        avg_time = float(entry_avg_time.get())
        broken_machines = int(entry_broken_machines.get())
        breakdown_time = float(entry_breakdown_time.get())
        lab_tat = float(entry_lab_TAT.get())
        no_of_machines = 6

        total_tests = samples * avg_tests
        total_time = total_tests * avg_time
        delay_due_to_breakdowns = (broken_machines * breakdown_time) / 60
        available_time = lab_tat * no_of_machines - delay_due_to_breakdowns
        delay_hours = max(0, (total_time - available_time) / no_of_machines)

        messagebox.showinfo("Delay Estimation", f"Your estimated maximum delay for a test is {delay_hours:.2f} hours")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numeric values.")

# Main window
root = tk.Tk()
root.title("OCD Simple Delay Predictor")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

# Styling
style = ttk.Style()
style.configure("TLabel", font=("Segoe UI", 11))
style.configure("TEntry", font=("Segoe UI", 11))
style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=6)

# Heading
heading = tk.Label(root, text="OCD Simple Delay Predictor", font=("Segoe UI", 18, "bold"), fg="#000000", bg="#f0f0f0")
heading.pack(pady=20)

# Frame for inputs
frame = ttk.Frame(root, padding=10)
frame.pack()

def add_field(row, label, entry_widget):
    ttk.Label(frame, text=label).grid(row=row, column=0, sticky='e', padx=10, pady=6)
    entry_widget.grid(row=row, column=1, padx=10, pady=6)

entry_samples = ttk.Entry(frame)
entry_avg_tests = ttk.Entry(frame)
entry_avg_tests.insert(0, "5")
entry_avg_time = ttk.Entry(frame)
entry_avg_time.insert(0, "1")
entry_broken_machines = ttk.Entry(frame)
entry_broken_machines.insert(0, "0")
entry_breakdown_time = ttk.Entry(frame)
entry_breakdown_time.insert(0, "0")
entry_lab_TAT = ttk.Entry(frame)
entry_lab_TAT.insert(0, "2.5")

fields = [
    ("Number of samples in machine:", entry_samples),
    ("Avg no. of tests per sample:", entry_avg_tests),
    ("Avg time per test (Hr):", entry_avg_time),
    ("No. of machines in breakdown:", entry_broken_machines),
    ("Breakdown time per machine (min):", entry_breakdown_time),
    ("Machine TAT (Hr):", entry_lab_TAT)
]

for i, (label, widget) in enumerate(fields):
    add_field(i, label, widget)

# Button
predict_btn = ttk.Button(root, text="Predict Delay", command=predict_delay)
predict_btn.pack(pady=20)

root.mainloop()
