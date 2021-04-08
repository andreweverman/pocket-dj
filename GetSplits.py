# Andrew Everman
# 1/26/20
# Goal of this file is to give the functionality to use spotify easily in another file
import subprocess
import time
from multiprocessing import Process
from threading import Thread
import json
import os
import sys
import numpy as np
import pickle
import librosa
import sklearn
import scipy
import time
import concurrent.futures
import urllib.request
from bs4 import BeautifulSoup
import youtube_dl
import platform
import SpotifyUtil

dir_path = os.path.dirname(os.path.realpath(__file__))


class GetSplits():

    def __init__(self, spotify=False, spotify_user=None, analysis_path='analysis', songs_path='songs',weights=[1, 1, 5, 5, 5, 5]):
        if spotify:
            if not spotify_user:
                self.spu = SpotifyUtil.SpotifyUtil()
            else:
                self.spu = SpotifyUtil.SpotifyUtil(spotify_user)

        self.dir_sep = '/'
        if platform.system() == 'Windows':
            self.dir_sep == '\\'

        self.folder_setup('analysis')
        self.folder_setup('songs')
        self.score_weights = np.array(weights)

        self.spotify_user = spotify_user

    def folder_setup(self, name):
        fp = '.' + self.dir_sep + name
        if not os.path.isdir(fp):
            try:
                print(name,"directory does not exist. Creating...")
                os.mkdir(fp)

            except OSError:
                print("Creation of the directory failed\n")
            else:
                print("Successfully created the ",name, "directory")

    # ------- Quick methods for clarity later on -----------

    def is_spotify_uri(self, uri):
        return uri.startswith('spotify:')

    def is_spotify_track(self, uri):
        return uri.startswith('spotify:track:') and len(uri) == 36

    def is_spotify_playlist(self, uri):
        return uri.startswith('spotify:playlist:') and len(uri) == 39

    def is_spotify_album(self, uri):
        return uri.startswith('spotify:album:') and len(uri) == 36

    # ------- A bunch of adapter methods -------
    def search_to_info(self, song):
        return self.spu.search_to_info(song)

    def get_playlist_track_uris(self, playlist_uri):
        return self.spu.get_playlist_track_uris(playlist_uri)

    def get_album_track_uris(self, album_uri):
        return self.spu.get_album_track_uris(album_uri)

    def get_track_name(self, uri):
        return self.spu.get_track_name(uri)

    def check_if_in_dir(self, uri, dirname):
        # removing extensions
        dl_songs = [self.just_uri_from_path(f) for f in os.listdir(dirname)]
        return uri in dl_songs

    def uri_to_p(self, uri):
        return uri.replace(':', '_')

    def p_to_uri(self, uri):
        return uri.replace('_', ':')

    def just_uri_from_path(self, fp):
        fp_split = fp.split(self.dir_sep)
        no_path = fp_split[len(fp_split)-1]
        no_extension = no_path[0:no_path.rfind('.')]
        uri = self.p_to_uri(no_extension)
        return uri

    # ------- Downloading and Pickling Logistics
    def load_analysis_for_uri(self, uri):
        with open('analysis' + self.dir_sep + self.uri_to_p(uri), 'rb') as infile:
            return pickle.load(infile)

    def load_analysis(self, song_info, spotify):
        '''
        The initial analysis that gets pickled is the bare section data
        from librosa

        This method then processes that initial information and then 
        analyzes that and if spotify is true, then adds
        on the additional spotify feature data. 


        song_info:  list    list of songs and their uris
                    song has 'uri' and 'name'

        spotify:    boolean whether or not to use the additional spotify information


        returns:    list of sections for a track.
                    each section has a bunch of info

        '''

        track_analysis = []
        for song in song_info:
            uri = song['uri']
            song_data = self.load_analysis_for_uri(uri)
            pruned, delta_split = self.prune_sections(song_data)
            sections = self.process_sections(song_data, pruned, delta_split)

            if spotify:
                features = self.spu.get_track_features(uri)
                keys = ['start', 'end']
                attrs = ['acousticness', 'danceability',
                         'energy', 'instrumentalness']
                for section in sections:
                    for key in keys:
                        for att in attrs:
                            section[key][att] = features[att]

            track_analysis.append(sections)
        return track_analysis

    def get_and_dump_analysis(self, aud_file):

        uri = self.just_uri_from_path(aud_file)
        analysis = self.get_initial_analysis(aud_file)
        with open('analysis' + self.dir_sep + self.uri_to_p(uri), 'wb') as outfile:
            pickle.dump(analysis, outfile)

    def download(self, song_info):
        # DOWNLOADING FILES

        processes = []
        for song in song_info:
            if song['uri'] == 'spotify:track:4gnmW2jAVmdDe6EGAGppFG':
                i = 0

            if not self.check_if_in_dir(song['uri'], 'songs'):

                time.sleep(1)
                t = Thread(target=self.download_from_q,
                           args=(song['name'], song['uri']))
                processes.append(t)
                t.start()

        for proc in processes:
            proc.join()

        print('All Downloaded')

    def download_from_q(self, q, uri, num=0):
        # TODO :FIX
        if self.check_if_in_dir(uri, 'songs'):
            return

        try:
            fp = "songs" + self.dir_sep + self.uri_to_p(uri)
            ydl_opts = {'outtmpl': fp}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:

                query = urllib.parse.quote(q)
                url = "https://www.youtube.com/results?search_query=" + query
                response = urllib.request.urlopen(url)
                html = response.read()
                soup = BeautifulSoup(html, 'html.parser')
                for i, vid in enumerate(soup.findAll(attrs={'class': 'yt-uix-tile-link'})):
                    link = vid['href']
                    if '&list' not in link and '&start_radio' not in link:
                        link = 'https://www.youtube.com' + link
                        ydl.download([link])                        
                        return

        except:
            # prevent infinite looping but give 10 of them a try
            num = i
            if num > 10:
                return
            self.download_from_q(q, uri, num+1)

    def get_song_info(self, songs):
        '''
        This will parse the songs that are put in for what they are
        Already done in db but this is for my own testing

        @param songs    list    A list of song queries, spotify playlists, and/or spotify tracks

        returns         list    A list of songs
                                Each song is a dictionary with name and uri
        '''
        song_info = []

        for song in songs:

            if self.is_spotify_uri(song):
                # going to see if it is a playlist or a track
                if self.is_spotify_track(song):
                    name = self.get_track_name(song)
                    uri = song
                    obj = {'name': name, 'uri': uri}
                    song_info.append(obj)
                elif self.is_spotify_playlist(song):
                    playlist_track_uris = self.get_playlist_track_uris(song)
                    song_info.extend(self.get_song_info(playlist_track_uris))
                    pass
                elif self.is_spotify_album(song):
                    album_track_uris = self.get_album_track_uris(song)
                    song_info.extend(self.get_song_info(album_track_uris))
                else:
                    # dont want to do anything if it is a messed up uri
                    pass
            else:
                # just a song name, lookup with spotify
                info = self.search_to_info(song)
                song_info.append(info)

        return song_info

    # ---------- The logic of getting the splits ----------

    def complete_get_splits(self, songs, spotify=True, pkl=False):
        '''
        The main method that should be called upon for what I am doing
        This will download the songs, get analysis, and spit out the splits

        Arguments:

        songs       list    query strings or spotify track uris
        spotify     bool    whether or not to use spotify feature data as additional information
        pkl         bool    whether or not to pickle the results of the analysis in a file


        returns     list    the complete splits
                            each split says the song and times to split


        '''

        # first getting songs and their spotify uris
        # input is allowed to be either so going to go through
        song_info = self.get_song_info(songs)

        # now that I have the songs, I will download them
        self.download(song_info)

        # now we will run analysis
        self.get_analysis(song_info, pkl)

        # analysis is stored in the files
        # will get out here only when all of them are downloaded
        track_analysis = self.load_analysis(song_info, spotify)

        splits = self.compute_splits(track_analysis, spotify)

        return splits

    def spotify_get_splits(self, songs, spotify=True, play=False):
        '''
        Counterpart to the complete_get_splits
        Gets the splits using spotify data only, no download

        Does same things overall as the complete_get_splits

        params:
        play: Whether or not to have spotify play the song
        '''
        song_info = self.get_song_info(songs)

        # song_info will have a spotify uri for each track in it,
        # so just going to grab those to pass into the spu.get_sections

        uris = [s['uri'] for s in song_info]

        analysis = self.spu.get_sections(uris, spotify=True)

        splits = self.compute_splits(analysis)

        if play and self.spotify_user != None:
            self.spu.play(splits)

    # ---------- Analysis section ---------------------

    def get_analysis(self, song_info, pkl=True):
        '''
        Multithreaded way to get all of the analysis for the songs that got input
        This will get the analysis and pickle it to the directory /analysis

        params:
        song_info   list    song names and spotify uris
        pkl         bool    wheter or not to save the data. Need to be true for now
        '''

        # TODO: LOOK AT THIS HSIT
        uris = [t['uri'] for t in song_info]
        analy_files = os.listdir(dir_path + self.dir_sep+"analysis")
        dl_files_with_ext = os.listdir(dir_path + self.dir_sep+"songs")
        dl_files_no_ext = [f.split('.')[0] for f in dl_files_with_ext]
        processes = []

        for uri in uris:
            track = self.uri_to_p(uri)

            index = dl_files_no_ext.index(track)
            track_file = dl_files_with_ext[index]
            song = "songs" + self.dir_sep + track_file

            if track not in analy_files:
                # need to find the actual file that it is downloaded under

                t = Thread(target=self.get_and_dump_analysis, args=(song,))
                processes.append(t)
                time.sleep(1)
                t.start()

        for proc in processes:
            proc.join()

    def get_initial_analysis(self, aud_file):

        y, sr = librosa.load(aud_file)

        # Saving librosa numpy array to file
        np.save(aud_file, y, allow_pickle=True)
        # Deleting audiofile, no longer need it
        if os.path.exists(aud_file):
            os.remove(aud_file)


        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7
        C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                                       bins_per_octave=BINS_PER_OCTAVE,
                                                       n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                    ref=np.max)

        S = librosa.stft(y)**2
        power = np.abs(S)**2
        p_mean = np.sum(power, axis=0, keepdims=True)
        # or whatever other reference power you want to use
        p_ref = np.max(power)
        loudness = librosa.power_to_db(p_mean, ref=p_ref)
        loudness = loudness.flatten()

        ##########################################################
        # To reduce dimensionality, we'll beat-synchronous the CQT
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)

        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                              sym=True)

        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        ##########################################################
        # And compute the balanced combination (Equations 6, 7, 9)

        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path

        L = scipy.sparse.csgraph.laplacian(A, normed=True)

        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k

        k = 5
        X = evecs[:, :k] / Cnorm[:, k-1:k]
        KM = sklearn.cluster.KMeans(n_clusters=k)
        seg_ids = KM.fit_predict(X)

        ###############################################################
        # Locate segment boundaries from the label sequence
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary

        bound_beats = bound_beats[bound_beats != beats.size]

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames,
                                               x_min=None,
                                               x_max=C.shape[1]-1)

        song_data = []
        uri = aud_file.split('/')
        uri = uri[len(uri)-1].split('.')[0]

        for i in range(len(bound_frames)):

            if i == 0:
                start = 0
                end = bound_frames[i]
                delta = 0

            elif i == len(bound_frames):
                start = bound_frames[i-1]
                end = loudness.shape[0]
            else:
                start = bound_frames[i-1]
                end = bound_frames[i]

        #     getting the average loudness for this section

            avg_l = loudness[start:end].mean()
            tempo = librosa.beat.tempo(y=y[start:end], sr=sr)[0]
            times = librosa.frames_to_time([start, end])
            start = times[0]
            end = times[1]

            if i != 0:
                delta = avg_l - song_data[i-1]['loudness']
            section = {'loudness': avg_l, 'time': times, 'delta': delta,
                       'tempo': tempo, 'uri': uri, 'path': aud_file}

        #     if section['loudness']> -40:
            song_data.append(section)

        return song_data

    def prune_sections(self, song_data):
        '''
        Given a song's section data, gets rid of some of the bad sections
        and gives better idea of where to cut sections off

        params:
        song_data:  list    the sections of the song

        returns: 
        good_sections:  the sections that are usable
        delta_split:    where the sections are good back to back
                        but should split between
                        Ex. change between section is too large but are both good
                        so split between
        '''
        # TODO: can make this cleaner but works for now

        loudnesses = np.array([sect['loudness'] for sect in song_data])
        loud_delta = np.array([sect['delta'] for sect in song_data])

        # any section loudness < -20 is for sure bad
        initial_quiet_prune = np.where(loudnesses > -40)[0]

        no_outliers_loudness = np.array(
            [loudnesses[val] for val in initial_quiet_prune])

        # want to use some stats to compute cutoff values for the loudness values
        average_loudness = np.mean(no_outliers_loudness)
        average_delta = np.mean(loud_delta)
        std_loudness = np.std(no_outliers_loudness)
        std_delta = np.std(loud_delta)

        # going to prune sections that are too quiet and are too quiet overall
        quiet_prune = np.where(loudnesses > average_loudness-std_loudness)[0]

        # We shouldnt actually remove if the delta is large here,
        # but we should have a cutoff for combining sections here

        # prune sections that have a very large change in volume
        delta_prune = np.where(loud_delta > average_delta-std_delta)[0]
        delta_split = np.where(loud_delta > average_delta+std_delta)[0] - 1
        delta_split = np.where(delta_split != 0)[0]

        # if two numbers are adjacent, then remove the section?

        good_sections = np.intersect1d(quiet_prune, delta_prune)

        return good_sections, delta_split

    def process_sections(self, song_data, good_sections, delta_split):

        # want to get the sections from the pruned that are adjacent
        potential_sections = []
        cut_index = 0
        for i in range(len(good_sections)):

            if i == len(good_sections)-1 or good_sections[i] != good_sections[i+1]-1 or good_sections[i] in delta_split:

                # get the tempo of the first and last section
                start_sec = song_data[good_sections[cut_index]]
                end_sec = song_data[good_sections[i]]
                if i == cut_index:
                    end_sec = start_sec.copy()

                start = start_sec['time'][0]
                end = end_sec['time'][1]
                duration = end-start

                start_sec['time'] = start
                end_sec['time'] = end

                # not adacent, cut here.
                covered_sections = list(good_sections[cut_index:i+1])
                info = {'sections': covered_sections, 'start': start_sec,
                        'end': end_sec, 'duration': duration, 'uri': start_sec['uri']}

                if duration>30:
                    potential_sections.append(info)

                # set cut start to next spot
                if not i == len(good_sections-1):
                    cut_index = i+1

        final_sections = []
        for section in potential_sections:
            if section['duration'] > 10:
                final_sections.append(section)

        return np.array(final_sections)

    # ---------- End Analysis Section ---------------------

    def compute_splits(self, track_analysis, spotify=False):
        '''
        This is for later use if the serverless stuff goes well

        This method is good for given that the songs are analyzed, you just call
        this and it will do the work

        Precondition  is that the analysis needed for this to work 
        is alredy computed and this just needs to load it
        '''

        # now have all info for multiple tracks
        # time to determine order and which sections to use
        matrix, all_sections, track_sections = self.build_matrix(
            track_analysis)

        splits = self.score_matrix(all_sections, matrix,  track_sections)

        return splits

    def build_matrix(self, info):
        '''
        Takes in the sections given to it by process_sections()
        Gives the best order and splits to use to play. 

        Returns array of track uris and the splits to use for it

        info => array of songs
        song => sections that we might want to
        section => duration, start/end =>  tempos, keys, and times

        returns:
        end_matrix:     nxn np array    The matrix that will be scored on 
        all_sections:   list            A flattened out list of every song and section
                                        used so each section has a number to the array  
        track_sections  list            All of the sections not flattened out? I honestly dont remember
        '''
        # things to check for:
        # 1. if the tempos are close
        # 2. if the loudness is close
        # 3. if the keys are close

        # need to get a flat section array made
        HIGH_VAL = 999

        i = 0
        all_sections = []
        track_sections = []
        for track in info:
            sections = []
            for section in track:

                all_sections.append(section)
                sections.append(i)
                i += 1
            track_sections.append(sections)

        all_sections = np.array(all_sections)

        section_count = len(all_sections)

        score_params = ['tempo', 'loudness', 'acousticness',
                        'danceability', 'energy', 'instrumentalness']
        

        # making 3d array

        # first dimension is so the score for a particular section is isolated
        # this should allow us to normalize the data and add weights for each section

        # the other two are the acutal score array dimensions

        matrix = np.zeros((len(score_params), section_count, section_count))

        # filling out scores for the matrix
        for i in range(section_count):

            first_section = all_sections[i]
            first_vals = [first_section['end'][x] for x in score_params]

            for j in range(section_count):

                second_section = all_sections[j]
                second_vals = [second_section['start'][x]
                               for x in score_params]

                # This is the actual score computation for each pair.
                # instead we can make arrays for each attribute, normalize them , and then combine them

                for k in range(len(score_params)):
                    if first_section['uri'] == second_section['uri']:
                        # from same track.  make this artificially high.
                        score = HIGH_VAL
                    else:
                        score = abs(round(first_vals[k] - second_vals[k], 3))
                    matrix[k][i][j] = score

        for i in range(matrix.shape[0]):
            matrix[i] *= self.score_weights[i]

        # this adds up the scores from the different metrics
        end_matrix = np.sum(matrix, axis=0)

        return end_matrix, all_sections, track_sections

    def score_matrix(self, all_sections,  matrix, track_sections):
        '''
        Given some 2d matrix that has the scores for having one song into the other, 
        get the optimal order to play these songs

        Row song leads into column song

        Currently a greedy algorithm.
        Finds the lowest score and puts that into the order. 
        Makes those songs unable to be used how they were useed again
        Ex. if one was start, makes it to where it wont pick it for that again

        Keeps doing this until each song is added to the order


        PARAMS:

        all_sections: flattened array of all of possible sections.
            they have the uri that they belong to and also can use track_sections to find where it is from
        matrix: score matrix that we will be going through
        track_sections: organizes which section number belongs to which track


        returns:    list    The ordered songs and sections to play each
        '''
        HIGH_VAL = 999
        order = []

        expected_song_count = 0
        # fixing the count that we should expect:
        for track in track_sections:
            if len(track) != 0:
                expected_song_count += 1

        # fixes issue with one song causing infinite looping
        if expected_song_count == 1:
            # only one song so skip the matrix. just play the longest section
            maximum = {'index': 0, 'duration': 0}
            for i, section in enumerate(all_sections):
                if section['duration'] > maximum['duration']:
                    maximum = {'index': i, 'duration': section['duration']}

            return np.array([all_sections[maximum['index']]])

        # look at last element of order, put in minumum value of song that is not in array already
        while len(order) < expected_song_count:

            indicies = np.where(matrix == matrix.min())
            if indicies[0].shape[0] > 1:
                indicies = indicies[0]
            start_ind = indicies[0].min()
            end_ind = indicies[1].min()

            if start_ind in order and end_ind in order:
                # means both values in already, this creates a paradox
                # so we will make value high and should be good
                matrix[start_ind][end_ind] = HIGH_VAL
            else:
                # making it artificially high to pick the contradicting pair
                matrix[end_ind][start_ind] = HIGH_VAL

                # need to make each section with track uris matching used ones above is taken out of the running
                for song in track_sections:
                    if start_ind in song:
                        for i in song:
                            if i != start_ind:
                                matrix[:, i] = HIGH_VAL
                            matrix[i] = HIGH_VAL
                    if end_ind in song:
                        for i in song:
                            if i != end_ind:
                                matrix[i] = HIGH_VAL
                            matrix[:, i] = HIGH_VAL

                # order array
                if len(order) == 0:
                    order.append(start_ind)
                    order.append(end_ind)
                else:
                    if end_ind in order:
                        index_end = order.index(end_ind)
                        order.insert(index_end, start_ind)
                    else:
                        # end song not in there already
                        if start_ind in order:
                            # add end song after start song
                            index_start = order.index(start_ind)
                            order.insert(index_start+1, end_ind)

                        else:
                            # neither song is already in there
                            # TODO: going to put this in at the front or end
                            # would be cool if we did some recursive way to check some differnt possibilities if we don't always just take the min

                            # check the matrix val for if second song is put at front vs first put at end
                            front_val = matrix[end_ind][order[0]]
                            end_val = matrix[order[len(order)-1]][start_ind]

                            min_val = min(front_val, end_val)

                            if front_val == min_val:
                                # add to front
                                order.insert(0, end_ind)
                                order.insert(0, start_ind)
                            else:
                                order.append(start_ind)
                                order.append(end_ind)

        # now have the order we want the songs.
        order = np.array(order)

        # reorders the songs accordingly
        splits = list(np.array(all_sections)[order])

        return splits

    
    