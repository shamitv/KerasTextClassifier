from pathlib import Path
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

src_data_dir='..'
corenlpDir='..'


pathlist = Path(src_data_dir).glob('**/*.xml')

for path in pathlist:
    file_path=str(path)
    print(file_path)
    tree = ET.parse(file_path)
    doc_elements=tree.findall(".//document")
    for de in doc_elements:
        url=de.attrib['url']
        text=de.text
        if(text is None):
           text=""
        else:
            soup = BeautifulSoup(text, 'html.parser')
            text=soup.get_text()
        print(url+"\t"+text)
