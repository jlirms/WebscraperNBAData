# -*- coding: utf-8 -*-
"""
Finding the distance which teams travelled in 2019

"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from time import sleep
from random import randint
from IPython.core.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import DistanceMetric

#%%
def getsoup(url):
  response = requests.get(url)
  return BeautifulSoup(response.content, 'html.parser')

def getTable_givenTbody(tablebody, division):
    '''this function simply scrapes the 2019 standings to find stats for each team'''
    new_table = pd.DataFrame()
    for row in tablebody.find_all('tr', class_ = lambda x: x !='thead'):
        column_marker = 0
        name = row.find('th').find('a')['href'][-13:-10]
        columns = row.find_all('td')
    
        new_table.loc[name,'fullname'] = row.find('th').find('a', href = True).text
        
        for column in columns:
            new_table.loc[name,column_marker] = column.get_text()
            column_marker += 1   
            
    return new_table

#%% Get the standings and each teaam full name from 2019
soup = getsoup("https://www.basketball-reference.com/leagues/NBA_2019.html")

eastbody = soup.find('table', id = 'confs_standings_E').find('tbody')
westbody = soup.find('table', id = 'confs_standings_W').find('tbody')

tmpeast = getTable_givenTbody(eastbody, 'E')
tmpwest = getTable_givenTbody(westbody, 'W')


#%%
def getgamesperteam(team, teamname):
    '''this function gets all the game locations for a given team'''
    
    soup = getsoup("https://www.basketball-reference.com/teams/" +team+ "/2019_games.html")
    table = soup.find('table',id = 'games').find('tbody')
    games = pd.DataFrame()
    
    for seq, row in enumerate(table.find_all('tr')): # class_=lambda x: x != 'thead')):
      ind = row.find('th').text
      if ind == 'G': ## some rows are blank for labels
        continue
      for stat in ['date_game','game_start_time','game_location','opp_name','game_result']:
          val = row.find('td', attrs = {"data-stat" : str(stat)}).text
          games.loc[ind, stat] = val
    
    games['TEAM'] = team
    games['date_game'] = games.date_game.apply(pd.Timestamp)
    games['date_month'] = games.date_game.dt.month
    
    def getlocation(at, opp, home): ## for apply function on games dataframe
      if at == '@':
        return opp ## away game
      else:
        return teamname ## home game, no travelling
        
    games['location'] = games.apply(lambda X: getlocation(X.game_location, X.opp_name, teamname), axis = 1)
    games['next_location'] = games.location.shift(-1)
    games.loc['82','next_location'] = teamname
    
    return games

#%% Get travel schedule for each team in both division

allteams_travel = []
for team, teamname in pd.concat([tmpeast,tmpwest]).fullname.items(): 
  print(team, teamname)
  allteams_travel.append(getgamesperteam(team, teamname))
  sleep(0.2)
  
#%%
dfdist = pd.concat(allteams_travel)
dfdist['TRIP'] = (dfdist.location != dfdist.next_location)
dfdist.reset_index(inplace = True, drop = False)
dfdist.rename(columns={'index':'gamenumber'}, inplace = True)



#%% Now we need to location of each teams stadium, scrape from Wikipedia
allarenas = getsoup('https://en.wikipedia.org/wiki/List_of_National_Basketball_Association_arenas')

arena_table = allarenas.find('tbody')
arena_rows = arena_table.find_all('tr')

dfarenas = pd.DataFrame()
for row in arena_rows[1:]: 
  allcells = row.find_all('a',href = True,  class_=lambda x: x != 'image')
  for e, cell in enumerate(allcells[:3]): 
    if e == 0:
      rowname = cell.text
      print(rowname)
      dfarenas.loc[rowname, e] = cell['href']
    else:
      dfarenas.loc[rowname, e] = cell.text

dfarenas.columns = ['URL','CITY','TEAM']
dfarenas.drop('Los Angeles Lakers', inplace = True) ## no Lakers stadium needed bc Clippers

#%% Get the latitude and longitude for each stadium, again from wikipedia

def getlat_long(URL):
  sleep(0.1)
  # print(URL)
  arena = getsoup('https://en.wikipedia.org' + URL)
  return arena.find('span', class_ = 'latitude').text , arena.find('span',class_ = 'longitude').text

dfarenas['latitude'], dfarenas['longitude'] = zip(*dfarenas.apply(lambda X: getlat_long(X.URL), axis = 1))

tarenas = dfarenas[['TEAM','latitude','longitude']].set_index('TEAM')
## extract the degree, minuts and seconds from lat and log measurements
dms2 = tarenas.latitude.str.extract(r'(?P<DEG>\d+)\D(?P<MIN>\d+)\D(?P<SEC>\d+)').astype(int) 
tarenas['lat'] = dms2.DEG + dms2.MIN/60 + dms2.SEC/(60**2) 
dms = tarenas.longitude.str.extract(r'(?P<DEG>\d+)\D(?P<MIN>\d+)\D(?P<SEC>\d+)').astype(int) 
tarenas['lon'] = -dms.DEG + dms.MIN/60 + dms.SEC/(60**2)

##convert into decimal degrees, and then into radians for python
tarenas['latr'] = np.radians(tarenas['lat'])
tarenas['lonr'] = np.radians(tarenas['lon'])


#%% Calculate distance from arenas using distance metric
## haversine functio: arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))
## finds distance between two locations given lat and log in radians

## sklearn was best option to use dist_pairwise 
## https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

dist = DistanceMetric.get_metric('haversine')
alldistances = dist.pairwise(tarenas[['latr','lonr']].values)*3798 
##miles because of NBA audience, 3798 is radius of earth in miles
alldistances = pd.DataFrame(alldistances, index = tarenas.index, columns = tarenas.index)

#%%Finishing touches, which team travelled the most in 2019?

def getdistances(a1, a2):
  if a1 == 'Los Angeles Lakers': a1 = 'Los Angeles Clippers'
  if a2 == 'Los Angeles Lakers': a2 = 'Los Angeles Clippers' ## same stadium
  return alldistances.loc[a1, a2]

dfdist['DISTANCE'] = dfdist[dfdist.TRIP == True].apply(lambda X: getdistances(X.location, X.next_location), axis = 1 )
dfdist.fillna(0, inplace = True)

stats = dfdist.groupby('TEAM')['DISTANCE'].sum().sort_values().to_frame()

stats['TOTALTRIPS'] = dfdist.groupby('TEAM')['TRIP'].sum()
stats = stats.sort_values('DISTANCE', ascending= False)
stats.DISTANCE = stats.DISTANCE.astype(int)
stats.TOTALTRIPS = stats.TOTALTRIPS.astype(int)

#%% Travel and arenas to flourish directy for visualization
dfdist.to_csv('ALL2019Travel.csv')
tarenas.to_csv('ArenaLocations.csv')
stats.to_csv('TRAVELstats.csv')