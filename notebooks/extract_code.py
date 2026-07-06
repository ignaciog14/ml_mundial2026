import json

with open('02_modelado_y_prediccion.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('extracted_nb.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            out.write(f"# CELL {i}\n")
            for line in cell['source']:
                out.write(line)
            out.write("\n\n")
