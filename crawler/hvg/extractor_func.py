# Seperate file needed to call multiprocessing.Pool on Windows...
import threading
from glob import glob
import pandas as pd
# from bs4 import BeautifulSoup
# import bs4
import html2text

    
def read_file(file):
    res = {}
    with open(file, 'r', encoding='utf8') as f:
        # title, description, tags, published_time, modified_time, section, content
        data = [line.strip() for line in f.readlines()]
    res['title'] = data[0]
    res['description'] = data[1]
    res['tags'] = data[2]
    res['published_time'] = data[3]
    res['modified_time'] = data[4]
    res['section'] = data[5]
    res['content'] = '\n'.join(data[6:])
    return res

def get_content(raw_content):
#     content = ''
#     html = BeautifulSoup(raw_content, 'html.parser')
#     if len([tag.name for tag in html.find_all()]) == 1:
#         content = html.text.strip()
#     else:
#         for p in html.find_all('p'):
#             content = content + '\n' + p.text.strip()
#     content = content.strip()
    
#     if content == "":
#         brs = []
#         for br in html.findAll('br'):
#             for sibling in [br.previous, br.next]:
#                 if isinstance(sibling, bs4.NavigableString):
#                     if sibling not in brs:
#                         brs.append(sibling)
#         content = ""
#         for br in brs:
#             text = br.strip()
#             if not text == '':
#                 content = content + '\n' + text
#         content = content.strip()
        
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_tables = True
    h.ignore_images = True
    h.ignore_emphasis = True
    h.escape_snob = True
    content = h.handle(raw_content)
    
    if content.endswith('(hvg.hu)'):
        content = content[:-len('(hvg.hu)')]
#     return content.replace('<br/>', '\n').strip()
    return content


def extract_file(file):
    data = read_file(file)
    raw_content = data['content']
    content = get_content(raw_content)
    return [data['title'],
            data['description'],
            data['tags'],
            data['published_time'],
            data['modified_time'],
            data['section'],
            content]