"""Local Tkinter UI. Suggested code is displayed, never executed."""
import argparse
import ast
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from gpt import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='model.npz')
    args = ap.parse_args()
    root = tk.Tk()
    root.title('NumPy GPT — experimental Python debugger')
    root.geometry('850x720')
    try:
        model = GPT.load(args.model)
    except (OSError, ValueError, KeyError) as e:
        messagebox.showerror('Model unavailable', f'Train a model first: python train.py\n\n{e}')
        root.destroy()
        return
    tk.Label(root, text='Small model trained from scratch. Suggestions may be incorrect. Review before use.').pack(pady=8)
    tk.Label(root, text='Paste a short Python function:').pack(anchor='w', padx=12)
    source = tk.Text(root, height=13, font=('Consolas', 11))
    source.pack(fill='both', expand=True, padx=12)
    source.insert('1.0', 'def add(a, b):\n    return a - b')
    status = tk.StringVar(value='Ready — inference stays on this computer.')
    tk.Label(root, textvariable=status).pack()
    output = tk.Text(root, height=14, font=('Consolas', 11), state='disabled')
    results = queue.Queue()
    def work(code):
        try:
            answer = model.generate(code, tokens=300)
            try:
                tree = ast.parse(answer)
                valid = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body)
            except (SyntaxError, ValueError):
                valid = False
            results.put((True, answer, valid))
        except Exception as e:
            results.put((False, str(e), False))
    def run():
        code = source.get('1.0', 'end-1c')
        button.config(state='disabled')
        status.set('Generating…')
        threading.Thread(target=work, args=(code,), daemon=True).start()
    button = tk.Button(root, text='Suggest a fix', command=run)
    button.pack(pady=6)
    output.pack(fill='both', expand=True, padx=12, pady=10)
    def poll():
        try:
            ok, text, valid = results.get_nowait()
            output.config(state='normal'); output.delete('1.0', 'end')
            output.insert('1.0', text); output.config(state='disabled')
            status.set(('Python parses; correctness has not been checked.' if valid else
                        'Invalid or missing Python function — model needs more training.') if ok else 'Unable to generate.')
            button.config(state='normal')
        except queue.Empty:
            pass
        root.after(100, poll)
    poll()
    root.mainloop()


if __name__ == '__main__':
    main()
