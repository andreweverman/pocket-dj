# Andrew Everman
# 1/26/20
# Goal of this file is to give the functionality to use spotify easily in another file

import spotipy
import json
import os
import sys
import spotipy.util as util
from spotipy import SpotifyClientCredentials
import numpy as np
import pickle

import time

dir_path = os.path.dirname(os.path.realpath(__file__))


class SpotifyUtil():

    def __init__(self, user_id=None):
        # getting spotify credentials
        if user_id:
            self.spotify = self.get_sp(user_id)
        else:
            # getting spotipy setup
            with open(dir_path + "/config.json", 'r') as config_file:
                config = json.load(config_file)

            # getting spotipy setup
            client_credentials_manager = SpotifyClientCredentials(
                client_id=config['client_id'], client_secret=config['client_secret'])
            self.spotify = spotipy.Spotify(
                client_credentials_manager=client_credentials_manager)

        # self.spotify = spotipy.Spotify()

    def search(self, q):

        return self.spotify.search(q, type='track')['tracks']['items'][0]

    def get_track_names(self,uris):
        lower = 0  
        limit = 50
        tracks = []
        while lower != len(uris):
            upper = lower+limit
            if upper >len(uris):
                upper = len(uris)
            section = uris[lower:upper]
            tracks.extend(self.spotify.tracks(section)['tracks'])
            lower = upper

        results = []
        for track in tracks:
            name = track['name']
            artist = track['artists'][0]['name']
            results.append(str(name) + " by " + str(artist))
        return results
    
    def get_track_name(self,uri):
        track =  self.spotify.track(uri)
        name = track['name']
        artist = track['artists'][0]['name']
        return (str(name) + " by " + str(artist))

    def get_sp(self, user_id):
        with open(dir_path + "/config.json", 'r') as config_file:
            config = json.load(config_file)

            # getting spotipy setup
            client_credentials_manager = SpotifyClientCredentials(
                client_id=config['client_id'], client_secret=config['client_secret'])
            scope = 'playlist-modify-public playlist-modify-private user-library-read user-library-modify user-read-playback-state user-modify-playback-state user-read-currently-playing'
            token = util.prompt_for_user_token(
                'spotify:user:'+str(user_id), scope, client_id=config['client_id'], client_secret=config['client_secret'], redirect_uri=config['redirect_uri'])
            if token:
                sp = spotipy.Spotify(
                    client_credentials_manager=client_credentials_manager, auth=token)
                sp.trace = False
            else:
                sys.exit()
            return sp

    def get_track_analysis(self, track_uris):
        '''
        Given array of track uris, get their info from spotify 
        @return: array with dict containing track name and the info below it
        '''
        analysis = []
        for track in track_uris:
            cur_track = self.spotify.audio_analysis(track)
            cur_track['uri'] = track
            analysis.append(cur_track)

        return analysis
    
    def get_track_features(self, track):
        '''
        Given array of track uris, get their info from spotify 
        @return: array with dict containing track name and the info below it
        '''
        
        return self.spotify.audio_features(track)[0]
        
    def get_playlist_track_uris(self, playlist_uri):
        
        limit = 100
        offset=0
        all_track_uris = []
        done = False
        while not done:
            playlist = self.spotify.playlist_tracks(playlist_uri,limit=100,offset=offset)
            tracks = playlist['items']
            all_track_uris.extend([t['track']['uri'] for t in tracks])
            done = playlist['next'] == None
            if not done:
                offset += limit
            
        
        return all_track_uris

    def get_album_track_uris(self,album_uri):
        tracks = self.spotify.album_tracks(album_uri)
        return [t['uri'] for t in tracks['items']]

    def search_to_info(self,song):
        track = self.search(song)
        name = track['name']
        artist = track['artists'][0]['name']
        full_name = (str(name) + " by " + str(artist))
        uri = track['uri']

        return {'name':full_name, 'uri':uri}
        
        

    def get_recommended(self, track_uris=[], artist_uris=[], limit=20):
        '''
        Get the spotify recommended songs giving them a list of track uris and artist uris
        @param arr_track_uris: list of track uris
        @return: array of songs that are similar with a call to the track info above so it will have all of the info and songs names/uris
        '''
        return self.spotify.recommendations(
            seed_artists=artist_uris, seed_tracks=track_uris, seed_genres=None, limit=limit)['tracks']

    def get_recommended_full(self, track_uris=[], artist_uris=[], limit=20):
        '''
        Get the spotify recommended songs giving them a list of track uris and artist uris
        @param arr_track_uris: list of track uris
        @return: array of songs that are similar with a call to the track info above so it will have all of the info and songs names/uris
        '''
        recommended = self.get_recommended(track_uris, artist_uris, limit)

        recommended_uris = []

        for track in recommended:
            recommended_uris.append(track['uri'])

        track_analysis = self.get_track_analysis(recommended_uris)
        return track_analysis

    def play(self, splits):

        self.print_results(splits)
        # info has the songs and
        devices = self.spotify.devices()['devices']

        device = ''
        
        for i,device in enumerate(devices):
            print(str(i) +': ' + str(device['name']))
         

        print('Which device would you like to use?')
        device_num = int(input())
        device = devices[device_num]['id']
        

        track_uris = []
        for track in splits:
            track_uris.append(track['uri'])

        self.spotify.start_playback(
            device_id=device, context_uri=None, uris=track_uris, offset=None)

        for track in splits:

            skip = int(track['start']['time'] * 1000)

            self.spotify.seek_track(skip, device)
            sleep = track['duration']
            time.sleep(sleep-1.5)

            self.spotify.next_track(device)
    
    def print_results(self,splits):
        print("\n")
        for song in splits:
            print("Playing", self.spotify.track(song['uri'])[
                    'name'], 'from', round(song['start']['time'],1), 'to', round(song['end']['time'],1))

    def q_to_splits(self, q):

        tracks = []
        for val in q:
            res = self.search(val)
            print(res['artists'][0]['name'], " - ", res['name'])
            tracks.append(res['uri'])

        return self.get_sections(tracks)

    def get_sections(self, track_uris,spotify=True):
        '''
        Goal of this function is to take in an array of songs and be able to give the times that should be split upon

        @param track_uris: array of track_uris that we will find the splits for
        returns: return an array of arrays. each element is represents a songs, and each songs will have a start and stop time 
        '''

        track_analysis = self.get_track_analysis(track_uris)
        track_sections = []

        # going through each track and trying to find the optimal parts to take from
        for track in track_analysis:
            uri = track['uri']


            sections = self.process_sections(
                track['sections'], track['uri'])
            if spotify:
                features = self.get_track_features(uri)
                keys = ['start', 'end']
                attrs = ['acousticness', 'danceability',
                         'energy', 'instrumentalness']
                for section in sections:
                    for key in keys:
                        for att in attrs:
                            section[key][att] = features[att]

            track_sections.append(sections)           
        
        return track_sections

    # TODO: normalize the data so the pruning is hopefully better.
    def prune_sections(self, sections):

        # TODO: can make this cleaner but works for now

        section_times = np.array(
            [[sect['start'], sect['start'] + sect['duration']] for sect in sections])

        loudnesses = np.array([sections[n]['loudness']
                               for n in range(0, len(sections))])
        loud_delta = np.insert(np.array([(round(loudnesses[n]-loudnesses[n-1], 1))
                                         for n in range(1, len(loudnesses))]), 0, 0)

        song_data = [{'times': section_times[i], 'delta': loud_delta[i],
                      'loudness': loudnesses[i]} for i in range(len(sections))]

        # any section loudness < -20 is for sure bad
        initial_quiet_prune = np.where(loudnesses > -20)[0]

        no_outliers_loudness = np.array(
            [loudnesses[val] for val in initial_quiet_prune])

        # want to use some stats to compute cutoff values for the loudness values
        average_loudness = np.mean(no_outliers_loudness)
        average_delta = np.mean(loud_delta)
        std_loudness = np.std(no_outliers_loudness)
        std_delta = np.std(loud_delta)

        # going to prune sections that are too quiet and are too quiet overall
        quiet_prune = np.where(loudnesses > average_loudness-std_loudness)[0]

        # TODO: think about this metric
        # prune sections that have a very large change in volume
        delta_prune = np.where(loud_delta > average_delta-std_delta)[0]
        # if two numbers are adjacent, then remove the section?

        good_sections = np.intersect1d(quiet_prune, delta_prune)

        return song_data, good_sections

    def process_sections(self, sections, uri):

        # TODO: Lets clean this method up
        # process_sections off of that

        # prune_sections gets info and prunes out the sections that are bad
        song_data, good_sections = self.prune_sections(sections)

        # want to get the sections from the pruned that are adjacent
        potential_sections = []
        cut_index = 0
        for i in range(len(good_sections)):

            if i == len(good_sections)-1 or good_sections[i] != good_sections[i+1]-1:

                # get the tempo of the first and last section
                start = song_data[good_sections[cut_index]]['times'][0]
                end = song_data[good_sections[i]]['times'][1]
                # not adacent, cut here.
                covered_sections = good_sections[cut_index:i+1]
                tempo_start = sections[covered_sections.min()]['tempo']
                tempo_end = sections[covered_sections.max()]['tempo']

                start_section = sections[covered_sections.min()]
                end_section = sections[covered_sections.max()]
                start_obj = {'time': start, 'tempo': tempo_start,
                             'loudness': start_section['loudness'], 'key': start_section['key']}
                end_obj = {'time': end, 'tempo': tempo_end,
                           'loudness': end_section['loudness'], 'key':  end_section['key']}
                info = {
                    'sections': list(covered_sections), 'start': start_obj, 'end': end_obj, 'duration': end-start, 'uri': uri}
                potential_sections.append(info)

                # set cut start to next spot
                if not i == len(good_sections-1):
                    cut_index = i+1

        return np.array(potential_sections)


