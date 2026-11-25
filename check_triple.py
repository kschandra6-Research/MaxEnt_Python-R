import pathlib
p = pathlib.Path("C:/Users/DuDu-PC/Downloads/MaxEnt_Python+R/app/python/raster_processor.py")
content = p.read_text()
import re
for m in re.finditer(r'"""', content):
    print(f'Triple quote at pos {m.start()}: {content[m.start():m.start()+50]}')
