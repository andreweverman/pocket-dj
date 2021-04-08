# pocket-dj

### This is a collection of just what my contribution to my Senoir Year Capstone Project. The final presentation video can be found [here](https://youtu.be/gB-wzreBj4U) and the final poster can be found [here](https://drive.google.com/file/d/167abQwcFU7wtf1SXopK3rlWxp1_wkhzR/view?usp=sharing). A quick summary of what my code did was take some list of songs and find the best ordering and general areas for songs to flow into each other.

## GetSplits
This directory is dedicated to getting the order and preiliminary split data. 

SpotifyUtil.py does similar things, but is not how we plan on moving forward.
SpotifyUtil.py is used to help get_splits, but is now not meant to be a complete thing of its own. 

### GetSplits.py 

This is the only file in this directory that is useful.

The overall flow is:
1. Get the track analysis from librosa and process that data
2. Build a scoring matrix for similarities for the given sections
3. Run the scoring matrix for minimum scores between sections and do that as a greedy algorithm.

#### Getting track analysis:
1. Get full track analysis from librosa
2. Prune sections from the above analysis that are too loud, change volume too much, etc. This gives better idea for where to split the larger sections
3. Process overall sections from the pruned ones. This will combine similar sections into larger ones so we have the longest section that makes sense

After this, we will have the general information for a song. There the built sections made of subsections and their start and end information. From this we will score these sections against each other to make the order. 

#### Building Scoring Matrix
Building the scoring matrix just takes the difference in scores for different attributes. The tempo and loudness is always used to score. The user has the option to enable spotify feature analysis, which includes the acousicness, danceability, energy and instrumentalness currently. In addition to taking the difference, there are weights applied to each attribute. The sections that are all from the same song have artifiially high scores so they will never get picked. 

#### Scoring the Matrix
The scoring just takes the minimum of the scoring matrix. Those two sections then are put into the order list. Then all contradicting sections are marked off limits by increasing score (Ex. if we pick section from song 'A', make all of its other sections unpickable so we dont play the same song twice, etc. ). This will choose the order for our songs. 


## Using GetSplits.py
When creating the object you have a few options.
For the best results I would use:

```gs = GetSplits(spotify=True, spotify_user='your_spotify_user_uri')```

This will let you use the spotify feature data and play on spotify.

The best method for an outer user to use is
#### Main Utility
```gs.complete_get_splits()```
This method will download and analayze the songs.
Then it will compute the splits.

Params: 
songs: list of:
* spotify track uri
* spotify album uri
* spotify playlist uri
* search query for a song
  
spotify: bool
    If true, then the splits will use additional info from spotify

pkl:    bool
    Doesn't acually work at the moment, I always pickle the analysis

The return is the ordered list of songs and where to split them.

#### Spotify
You can also use ```gs.spotify_get_splits()```
It is the same end result as ```gs.complete_get_splits()```, 
but it uses spotify's section data

Very similar params to ```gs.complete_get_splits()```

Params: 
songs: list of:
* spotify track uri
* spotify album uri
* spotify playlist uri
* search query for a song
  
spotify: bool
    If true, then the splits will use additional info from spotify

play:    bool
    Plays on spotify

The return is the ordered list of songs and where to split them.




