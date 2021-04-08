import GetSplits as GS
from pymongo import MongoClient
import random
import sys
import json
import argparse


client = MongoClient("")
db = client.party

'''
Args 
1. [Partycode]
2. tempo
3. loudness
4. acousticness
5. danceability
6. energy
7. instrumentalness
'''

argp = argparse.ArgumentParser()
argp.add_argument('-p', '--partycode', help='code to get into an ongoing party', required=True)
argp.add_argument('-t', '--tempo', help='beats per minute of a song', default=1)
argp.add_argument('-l', '--loudness', help='measure of how loud a song is', default=1)
argp.add_argument('-a', '--acousticness', help='measure of how much real instruments', default=5)
argp.add_argument('-d', '--danceability', help='measure of how danceable a song is', default=5)
argp.add_argument('-e', '--energy', help='measure of the average energy of a song', default=5)
argp.add_argument('-i', '--instumentalness', help='measure of lack of vocals', default=5)
args = argp.parse_args()

partycode = args.partycode

parties = db['Parties']
party = parties.find_one({'partyCode': int(partycode)})

requests = [req['uri'] for req in party['requests']]

requests = list(dict.fromkeys(requests))

weights = [int(w) for w in [args.tempo, args.loudness, args.acousticness, args.danceability, args.energy, args.instumentalness]]

gs = GS.GetSplits(spotify=True, weights=weights)

splits = gs.complete_get_splits(requests)

for split in splits:
    
    split['sections'] = [int(section) for section in split['sections']]
    for tag in ['start','end']:
        needs_convert = ['loudness','delta']
        for part in needs_convert:
            split[tag][part] = float(split[tag][part])

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(list(splits), f, ensure_ascii=False, indent=4)