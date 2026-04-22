import tkinter as tk
import stanza

stanza.download('mr')
nlp = stanza.Pipeline(lang='mr', processors='tokenize,pos')

def tag_text():
    text = entry.get()
    doc = nlp(text)
    
    result.delete("1.0", tk.END)
    
    for sentence in doc.sentences:
        for word in sentence.words:
            result.insert(tk.END, f"{word.text} --> {word.upos}\n")

root = tk.Tk()
root.title("Marathi POS Tagger")

entry = tk.Entry(root, width=50)
entry.pack()

btn = tk.Button(root, text="Tag POS", command=tag_text)
btn.pack()

result = tk.Text(root, height=15, width=50)
result.pack()

root.mainloop()