# heostarpiano
Every script used for maintaining a piano Youtube channel

# Environment Setup
To run the scripts in this repo, make sure to execute the following to download dependencies.
- On Windows
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```
- On Mac/Linux
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
And then
    ```
    pip install -r requirements.txt
    ```

# Running Synthesia.py
To build a Synthesia style note video out of a midi file, first convert your midi file into json format. 
https://www.visipiano.com/midi-to-json-converter/

Now you are ready to run the script. Script can run in two modes.
1. Production Mode
In this mode, the script will generate ordinary Syntehsia style video.
Below is the example command you can run from the project root
Assuming that you have activated the virtual environment by ```source venv/bin/activate```
```
python scripts/synthesia.py -p -m "/path/to/your/midi/json/file" -b 140 -x 3.0 -y "https://www.youtube.com/thebackgroundvideo/"
```
To explain the options,
```
-p: Production mode
-m: Path to the midi json file
-b: BPM (To determine long note lengths)
-x: Note Speed
-y: (optional) Background video you want to add
```

2. Tutorial Mode
In this mode, you can create a tutorial video that prints out key names for each note.
You can also make write a finger number for each note when you run this mode.
Below is the example command. Assuming that you have activated the virtual environment by ```source venv/bin/activate```
```
python scripts/synthesia.py -t -m "/path/to/your/midi/json/file" -c 1 -s "/path/to/your/tutorial/script"
```
To exaplain the options,
```
-t: Tutorial mode
-m: Path to the midi json file
-c: code major (1) or minor (0) for key name printout
-s: Path to the tutorial text script. Look for tutorial_script_example.txt for more details.
```

# Running tag_generator.py
This is a auto tag generator for youtube video in Heostar channel.
Simply follow the instruction by running the script ```python scripts/tag_generator.py```
It will copy the tags to your computer clipboard, and you can simply paste it.