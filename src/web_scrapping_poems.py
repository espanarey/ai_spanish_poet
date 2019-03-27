# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:34:57 2019

@author: reynaldo.espana.rey

Web scrapping algorithm to build data set for text generator

source: https://towardsdatascience.com/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460

"""

# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd
import requests
import re
import time
import os
from bs4 import BeautifulSoup
import string



# =============================================================================
# Functions
# =============================================================================
# request page and make it BeautifulSoup
def get_page(url, verbose=0):
    # get page
    response = requests.get(url)
    if verbose:
        print('Successful:', str(response) =='<Response [200]>')
    if str(response) =='<Response [200]>':
        # BeautifulSoup data structure
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    return str(response)

# function to retrieve links from inspector gadget pluggin
def get_href(url, attr):
    # get page
    soup = get_page(url)
    # get data links
    data = soup.select(attr)
    links = np.unique([x['href'] for x in data])
    return links

def get_text(url, attr):
    # get page
    soup = get_page(url)
    # get data links
    data = soup.select(attr)
    return data

# valid file name
def valid_name(value):
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value

# funtion to remove chars
def remove_chars(doc, chars_2remove=None):
    if chars_2remove is None:
        # list of character not UTF-8 to be remove from doc
        chars_2remove = ['\x85', '\x91', '\x92', '\x93', '\x94', '\x96', 
                         '\x97', '\xa0']
    # as reggex expression
    chars_2remove = '[' + ''.join(chars_2remove) + ']'
    # erase 
    doc = re.sub(chars_2remove, ' ', doc)
    doc = re.sub(' +', ' ', doc).strip()
    return doc


# =============================================================================
# Spanish poems
# =============================================================================
#### Spanish poems web page source
# root source
url_root = 'https://www.poemas-del-alma.com/'

## Path to use
## Retrieve poems and save it in .txt
path_poems = '../data/DB/spanish poems/'
# save list of poems links
path_poems_links = '../data/DB/poems_list.csv'



# =============================================================================
# Poems
# =============================================================================

##### POETS #####
# poems by author in alphabetial order
alphabet = [x for x in string.ascii_uppercase]

# get list of poets
poets = pd.DataFrame()
for letter in alphabet:
    print(letter)
    links = get_href(url_root + letter + '.html', attr='#content li a')
    authors = pd.DataFrame({'author': [x.split('/')[-1].split('.')[0] for x in links],
                            'link': links})
    poets = poets.append(authors)
    time.sleep(.5)
poets = poets.reset_index(drop=True)
print('Poests found:', len(poets))



##### POEMS #####
# go throgh all the poems in poets
# run only for poems not already in folder
poems = pd.read_csv(path_poems_links)
# filter poets to scrap
poets['in_disk'] = poets['author'].isin(poems['author'])
# filter songs df
print ('Files in disk already:', poets.groupby(['in_disk']).size())

# loop to remaining poets
poets_2scrap = poets[poets['in_disk']==False]
# shuffle, else all errors will be first
poets_2scrap = poets_2scrap.sample(frac=1).reset_index(drop=True)

# loop for each poet link
for index, row in poets_2scrap.iterrows():
    if (index % 25 == 0): 
        print('\n\n- Progress %:', index/len(poets_2scrap), '- Total poems:', len(poems))
        time.sleep(5)
    try:
        # get page with poems links
        links = get_href(row['link'], attr='#block-poems a')
        time.sleep(.5)
        links = pd.DataFrame({'poem': links})
        # save and append
        links['author'] = row['author']
        links['author_link'] = row['link']
        poems = poems.append(links, sort=False)
    except:
        print("An exception occurred:", row['link'])
        time.sleep(30)
print('Poems found:', len(poems))
poems.to_csv(path_poems_links, index=False)
    
    

# =============================================================================
# COURPUS
# =============================================================================

### Create poem corpus and save it as .txt
# list of poems to search
poems = pd.read_csv(path_poems_links)
print('Poems found:', len(poems))

# run only for poems not already in folder
# get file names
poems_files = os.listdir(path_poems)
# get ids of song in disk
poems_files_ids = [x.split('.')[0] for x in poems_files]
# filter poems df
poems['id'] = [x.split('.')[0] for x in poems['poem']]
poems['in_disk'] = poems['id'].isin(poems_files_ids)
print ('Files in disk already:', poems.groupby(['in_disk']).size())
# filter files to run webscrappin
poems_2scrap = poems[poems['in_disk']==False]
# shuffle, else all errors will be first
poems_2scrap = poems_2scrap.sample(frac=1).reset_index(drop=True)



# keep count of errors
errors = 0
# loop for each poet link
for index, row in poems_2scrap.iterrows():
    if (index % 20 == 0): 
        counter = len(poems[poems['in_disk']==True])+index        
        print('\n\n- Progress %: {0:.4f}'.format(index/len(poems_2scrap)),               
              '- Poems in disk: {}'.format(counter),
              '- Total %: {0:.4f}'.format(counter/len(poems)))
    try:
        # get page
        link = row['poem']
        soup = get_page(url_root + link)  
        # wait 1 second to not overheat the webscraping
        time.sleep(.5)
        # get poem
        page = soup.select('#contentfont p')
        if len(page):
                doc = str()
                for x in page:
                    doc = doc + x.getText()
                # encoding
                doc = remove_chars(doc)
                # TODO: remove chars                
                # save file
                filename = link.split('.')[0]
                with open(path_poems + filename + '.txt', "w") as text_file:
                    text_file.write(doc)
        else:
            print(link, '- poem is a set of poems')
            # get links
            links = get_href(url_root + link, attr='.list-poems a')
            #time.sleep(.5)
            links = pd.DataFrame({'poem': links})
            # save and append
            links['author'] = row['author']
            links['author_link'] = row['author_link']
            # update csv
            print('adding:', len(links), 'poems to .csv list')    
            poems = poems.append(links, sort=False)
            poems.to_csv(path_poems_links, index=False)
    except:
        errors+=1 
        print("An exception occurred:", link, '- error:', errors)
        if errors > 99:
            break # more than 99 errors stop websracping
        if (errors % 15 == 0):
            time.sleep(60) # wait 1 minute
        else: 
            time.sleep(5)
    
    



# =============================================================================
# Spanish love poems
# =============================================================================
#### Source web page
url_root = 'https://www.poemas-del-alma.com/'
# first page
url_1 = 'amor.htm'
## Path to use
## Retrieve poems and save it in .txt
path_poems = '../data/DB/love poems/'




# Get pagination
pages = get_href(url_root + url_1, attr='.btn-blue')
# append first page
pages = np.append(pages, [url_1])

# loop for all pages and get links for poems doc
poems = []
for link in pages:
    print(link)
    poems_page = get_href(url_root + link, attr='.list-poems a')
    poems.extend(poems_page)
print('Poems found:', len(poems))


# get doc for all poems and save it in .txt files
for link in poems:
    print(link)
    # get page
    soup = get_page(url_root + link)  
    # wait 1 second to not overheat the webscraping
    time.sleep(.5)
    # get poem
    page = soup.select('#contentfont p')
    if len(page):
            doc = str()
            for x in page:
                doc = doc + x.getText()
            # encoding
            doc = remove_chars(doc)
            # TODO: remove chars                
            # save file
            filename = link.split('.')[0]
            with open(path_poems + filename + '.txt', "w") as text_file:
                text_file.write(doc)
    else:
        print("An exception occurred:", link)


'''        
        print(link, '- poem is a set of poems')
        # get links
        links = get_href(url_root + link, attr='.list-poems a')
        time.sleep(.5)
        for link_2 in links:
            print(link_2)
            # get page
            soup = get_page(url_root + link_2)  
            # wait 1 second to not overheat the webscraping
            time.sleep(.5)
            # get poem
            page = soup.select('.block-poem-entry')
            if len(page):
                doc = str()
                for x in page:
                    doc = doc + x.getText()
                # encoding
                doc = remove_chars(doc)
                # TODO: remove chars                
                # save file
                filename = link_2.split('.')[0]
                with open(path_poems + filename + '.txt', "w") as text_file:
                    text_file.write(doc)
'''                

    
    
    
    
    