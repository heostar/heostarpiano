from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, TextClip, ImageSequenceClip, concatenate_videoclips
from moviepy.video.fx import resize
import json
import argparse
import math
import musicsheet
import cv2
import numpy as np
import yt_dlp
import os

class Synthesia:
    isWhiteKey = [True, False, True, False, True, True, False, True, False, True, False, True]
    keyNamesKoreanSharp = ["도","도#","레","레#","미","파","파#","솔","솔#","라","라#","시"]
    keyNamesEnglishSharp = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    keyNamesKoreanFlat = ["도","레b","레","미b","미","파","솔b","솔","라b","라","시b","시"]
    keyNamesEnglishFlat = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
    # Bpm
    # LongNoteThreshold
    # FallingNotesSpeed
    # FallingNotesXs
    # IsWhiteKeyArray

    def __init__(self, bpm, fallingNotesSpeed, fallingHeight, defaultSpeed, midi_path):
        self.FallingHeight = fallingHeight
        self.DefaultSpeed = defaultSpeed
        self.Bpm = bpm
        self.LongNoteThreshold = 60.0/bpm*1.5 # in seconds
        self.FallingNotesSpeed = fallingNotesSpeed
        self.MidiPath = midi_path

        # Some constants
        KeyMiddlesInOctave = [18.5, 34, 55.5, 77, 92.5, 129.5, 142.5, 166.5, 185, 203.5, 227, 240.5]
        KeyWalker = 9; #Cause piano keyboard starts from "la"
        KeyOctaveCounter = -1
        OctaveWidth=258.6

        # Global variables for falling notes and white/black key
        self.FallingNotesXs = []
        self.IsWhiteKeyArray = []
        for i in range(88):
            layer_x = (37*2 - 1) + KeyOctaveCounter*OctaveWidth + KeyMiddlesInOctave[KeyWalker]
            self.FallingNotesXs.append(layer_x)
            if self.isWhiteKey[(i+9)%12]:
                self.IsWhiteKeyArray.append(True)
            else:
                self.IsWhiteKeyArray.append(False)

            KeyWalker = KeyWalker + 1
            if KeyWalker >= 12:
                KeyWalker = 0
                KeyOctaveCounter = KeyOctaveCounter + 1

    def create_key(self, key_index, isActive, starttime, duration, hand=0):
        # this function is used for tutorial only
        image_list = ["white1","black1","white2","black1","white3","white4","black1","white5","black1","white6","black1","white7"]
        
        left_margins = [2, 25, 40, 68, 77, 113, 133, 150, 175, 187, 218, 224]
        starting_margin = -185
        octave_margin = 258
        handColors = ["", "L", "R"]

        key = (key_index+9)%12
        octave = math.floor((key_index+9)/12)
        image_name = image_list[key]
        if (octave == 0) and (key == 9):
            image_name = "white6_0"
        elif (octave == 8):
            image_name = "white1_0"
        if isActive:
            # Create mask
            mask_name = "assets/mask_"+image_name+".png"
            mask_image = ImageClip(mask_name, ismask=True)
            mask_clip = mask_image.to_mask()

            image_name = "assets/active"+handColors[hand]+"_"+image_name+".mp4"
            key_layer = VideoFileClip(image_name)
            key_layer = key_layer.set_start(starttime - 0.25)
            key_layer = key_layer.set_position((starting_margin + octave_margin*octave + left_margins[key], 886))
            key_layer = key_layer.set_mask(mask_clip)
            key_layer.fps = 60
            return key_layer
        else:
            image_name = "assets/inactive_"+image_name+".png"
        key_image = ImageClip(image_name, duration=duration)
        key_image = key_image.set_start(starttime)
        key_layer = key_image.set_position((starting_margin + octave_margin*octave + left_margins[key], 886))
        key_layer.fps=60

        return key_layer
    
    def create_finger_under_key(self, key_index, starttime, finger):
        left_margins = [2, 25, 40, 68, 77, 113, 133, 150, 175, 187, 218, 224]
        starting_margin = -185
        octave_margin = 258

        key = (key_index+9)%12
        octave = math.floor((key_index+9)/12)

        video_name = "assets/finger_number" + finger + ".mp4"
        finger_layer = VideoFileClip(video_name)
        finger_layer = finger_layer.set_start(starttime - 1.25)
        finger_layer = finger_layer.set_position((starting_margin + octave_margin*octave + left_margins[key], 886 - 40))
        finger_layer.fps = 60

        return finger_layer


    def create_full_keyboard_inactive(self, clip_duration):
        layers_to_return = []
        
        background_image = ImageClip("assets/keyboard_background.png", duration=clip_duration)
        background_layer = background_image.set_start(0)
        background_layer = background_layer.set_position((0, 886))
        layers_to_return.append(background_layer)

        starting_margin = -185
        octave_margin = 258
        for i in range(88):
            # For every octave, add vertical lines
            is_octave = ((i+9)%12 == 0)
            if is_octave:
                octave = math.floor((i+9)/12)
                left_margin = starting_margin + octave_margin*octave
                octaveline_image = ImageClip("assets/octave_line.png", duration=clip_duration)
                octaveline_layer = octaveline_image.set_start(0)
                octaveline_layer = octaveline_layer.set_position((left_margin, 0))
                layers_to_return.append(octaveline_layer)

            key_layer = self.create_key(i, False, 0, clip_duration)
            layers_to_return.append(key_layer)
        return layers_to_return

    # function that returns the image clip of a note
    # input: note hit time , duration
    # output: Image Clip
    def create_note_clip(self, notenum, hittime, duration, forTutorial=False, isCodeMajor=True, hand=0, finger=None):

        black_note_images = ["note_black.png", "LH_B_note.png", "RH_B_note.png"]
        white_note_images = ["note.png", "LH_W_note.png", "RH_W_note.png"]

        # create image clip
        image = None
        textclip = None
        fingerclip = None
        noteMelodyStr = ""
        if notenum == -100:
            pedal_image_name = 'assets/pedal60.png' # for production
            if forTutorial:
                pedal_image_name = 'assets/pedal.png' # for tutorial
            #Sustain pedal
            image = ImageClip(pedal_image_name, duration=hittime + duration + (1080 - self.FallingHeight - 5)/self.DefaultSpeed)
        elif self.isWhiteKey[(notenum+9)%12]:
            image = ImageClip("assets/" + white_note_images[hand], duration=hittime + duration + (1080 - self.FallingHeight - 32)/self.DefaultSpeed)
        else:
            image = ImageClip("assets/" + black_note_images[hand], duration=hittime + duration + (1080 - self.FallingHeight - 22)/self.DefaultSpeed)
        if forTutorial:
            keyIndex = (notenum+9)%12
            if isCodeMajor:
                noteMelodyStr = self.keyNamesKoreanSharp[keyIndex]# + "("+self.keyNamesEnglishSharp[keyIndex]+")"
            else:
                noteMelodyStr = self.keyNamesKoreanFlat[keyIndex]# + "("+self.keyNamesEnglishFlat[keyIndex]+")"
            fontSize = 35
            textclip = TextClip(noteMelodyStr, fontsize=fontSize, color='white', font='MapoGoldenPier')
            textclip = textclip.set_duration(hittime + duration + (1080 - self.FallingHeight)/self.DefaultSpeed)
            if finger != None:
                fingerclip = TextClip(finger, fontsize=fontSize, color='white', font='MapoGoldenPier')
                fingerclip = fingerclip.set_duration(hittime + duration + (1080 - self.FallingHeight)/self.DefaultSpeed)
        image_h = image.h
        image_w = image.w

        def get_quadratic_lambda(v1_obj, v2_obj, td, yd, note_height_lambda):
            A = (v2_obj['v'] - v1_obj['v']) / (v2_obj['t'] - v1_obj['t'])
            B = (v1_obj['v'])*(v1_obj['v'] - v2_obj['v'])/(v2_obj['t'] - v1_obj['t'])
            D = lambda Ca, Cb, t : 1/2*Ca*t*t + Cb*t
            C = yd - D(A,B,td) + D(A,B,v1_obj['t'])
            
            return lambda t : D(A,B,t) - D(A,B,v1_obj['t']) + C - note_height_lambda(t)
        
        def get_linear_lambda(v_obj, td, yd, image_h):
            return lambda t : v_obj['v']*(t - td) + yd - image_h
        
        def get_velocity_at_time(t):
            _v1_obj = None
            _v2_obj = None
            for i in range(len(self.FallingNotesSpeed)):
                if self.FallingNotesSpeed[i]['t'] <= t:
                    if len(self.FallingNotesSpeed) > i + 1:
                        if t <= self.FallingNotesSpeed[i+1]['t']:
                            _v1_obj = self.FallingNotesSpeed[i]
                            _v2_obj = self.FallingNotesSpeed[i+1]
            
            if _v1_obj != None:
                return (_v2_obj['v'] - _v1_obj['v'])/(_v2_obj['t'] - _v1_obj['t'])*(t - _v1_obj['t']) + _v1_obj['v']
            else:
                return None
            

        def get_lambdas(v1Obj, v2Obj, tKnown, yKnown, isLongNote):#hittime, FallingHeight
            dictLambdas = {'start':v1Obj['t'], 'end':v2Obj['t']}
            v1Y = None

            if isLongNote: # if it is long note
                lambdaLongNoteHeight = lambda t : ((v2Obj['v'] - v1Obj['v'])/(v2Obj['t'] - v1Obj['t'])*(t - v1Obj['t']) + v1Obj['v'])*duration
                if v1Obj['v'] != v2Obj['v']: # Quadratic
                    # Top Y                    
                    dictLambdas['getYtop'] = get_quadratic_lambda(v1Obj, v2Obj, tKnown, yKnown+1, lambda t : image_h/2 + lambdaLongNoteHeight(t))
                    # Middle Y
                    dictLambdas['getYmiddle'] = get_quadratic_lambda(v1Obj, v2Obj, tKnown, yKnown, lambdaLongNoteHeight)
                    # Middle H
                    hLambda = lambda t: (image_w+2, lambdaLongNoteHeight(t))
                    dictLambdas['getHmiddle'] = hLambda
                    # Bottom piece
                    dictLambdas['getYbottom'] = get_quadratic_lambda(v1Obj, v2Obj, tKnown, yKnown, lambda t : 0)
                else: # Linear
                    longNoteHeight = lambdaLongNoteHeight(v1Obj['t'])
                    # Top piece
                    dictLambdas['getYtop'] = get_linear_lambda(v1Obj, tKnown, yKnown+2, longNoteHeight + image_h/2)
                    # Middle piece
                    dictLambdas['getYmiddle'] = get_linear_lambda(v1Obj, tKnown, yKnown+1, longNoteHeight)
                    # Middle H
                    hLambda = lambda t: (image_w+2, longNoteHeight)
                    dictLambdas['getHmiddle'] = hLambda
                    # Bottom piece
                    dictLambdas['getYbottom'] = get_linear_lambda(v1Obj, tKnown, yKnown, 0)
                
                v1Y = dictLambdas['getYbottom'](v1Obj['t'])

            else: # regular short note
                if v1Obj['v'] != v2Obj['v']: # Quadratic
                    dictLambdas['getY'] = get_quadratic_lambda(v1Obj, v2Obj, tKnown, yKnown, lambda t : image_h/2)
                else: # Linear
                    dictLambdas['getY'] = get_linear_lambda(v1Obj, tKnown, yKnown, image_h/2)                
                
                v1Y = dictLambdas['getY'](v1Obj['t'])
            
            return dictLambdas, v1Obj['t'], v1Y
        
        isLongNote = False
        if (forTutorial == False) and (duration > self.LongNoteThreshold):
            isLongNote = True
            # why do we need note vanishing time ?? okay, to give a starting point for searching the right interval
        
        tLambdas = [] # [{'start':startting time, 'end':end time, 'getY':lambda function for Y, ...}, ... ]
        tKnown = hittime
        yKnown = self.FallingHeight

        for i in range(len(self.FallingNotesSpeed)):
            i_current = len(self.FallingNotesSpeed) - i - 1
            i_next = i_current - 1

            v2_obj = self.FallingNotesSpeed[i_current]
            if i_next >= 0 and yKnown >= 0:
                v1_obj = self.FallingNotesSpeed[i_next]
                if v1_obj['t'] <= tKnown and tKnown <= v2_obj['t']:
                    #within the range, acquire the equation
                    noteLambdas, tKnown, yKnown = get_lambdas(v1_obj, v2_obj, tKnown, yKnown, isLongNote)
                    tLambdas.insert(0, noteLambdas)
            else: # last item
                break

        def get_key_lambda(key):
            default_value = -200
            if key == 'getHmiddle':
                default_value = (image.w, 1)
            def key_lambda(t):
                if t < tLambdas[0]['start']:
                    return default_value
                for i in range(len(tLambdas)):
                    if t <= tLambdas[i]['end']:
                        return tLambdas[i][key](t)
                return default_value
                
            return key_lambda

        # fill in details
        x_pos = 0
        if notenum >= 0:
            x_pos = self.FallingNotesXs[notenum] - image_w/2
        if 'getYtop' in tLambdas[0]:
            # long note
            yTopLambda = get_key_lambda('getYtop')
            yMiddleLambda = get_key_lambda('getYmiddle')
            yBottomLambda = get_key_lambda('getYbottom')
            hMiddleLambda = get_key_lambda('getHmiddle')

            top_half = image.crop(y1=0, y2=image_h/2-1).set_position(lambda t: (x_pos, yTopLambda(t)))
            top_half.fps = 60

            stretch_layer = image.crop(y1=image_h/2-1, y2=image_h/2+1).resize(lambda t: hMiddleLambda(t)).set_position(lambda t: (x_pos - 1, yMiddleLambda(t)))
            stretch_layer.fps = 60
            
            bottom_half = image.crop(y1=image_h/2+1, y2=image_h).set_position(lambda t: (x_pos, yBottomLambda(t)))
            bottom_half.fps = 60

            return [top_half, stretch_layer, bottom_half]
        else:
            # short note (plus possible tutorial mode)
            layersToReturn = []

            yLambda = get_key_lambda('getY')

            shortnote_layer = image.set_position(lambda t: (x_pos, yLambda(t)))            
            shortnote_layer.fps = 60
            if notenum <= -100:
                shortnote_layer.set_opacity(0)
            elif forTutorial:
                x_text_pos = self.FallingNotesXs[notenum] - textclip.size[0]/2
                if x_text_pos < 0:
                    x_text_pos = 0
                if x_text_pos + textclip.size[0] >= 1920:
                    x_text_pos = 1920 - textclip.size[0]
                notetext_layer = textclip.set_position(lambda t: (x_text_pos, yLambda(t) - textclip.size[1] - 10))
                layersToReturn.append(notetext_layer)

                if finger != None:
                    x_finger_pos = self.FallingNotesXs[notenum] - fingerclip.size[0]/2
                    if x_finger_pos < 0:
                        x_finger_pos = 0
                    if x_finger_pos + fingerclip.size[0] >= 1920:
                        x_finger_pos = 1920 - fingerclip.size[0]
                    notefinger_layer = fingerclip.set_position(lambda t: (x_finger_pos, yLambda(t) + fingerclip.size[1] + 2))
                    layersToReturn.append(notefinger_layer)

            layersToReturn.append(shortnote_layer)
            return layersToReturn
    
    # Step 1: Download a video from a YouTube link
    def download_youtube_video(self, youtube_url, output_path='video.mp4'):
        print("Youtube url right here too", youtube_url)
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': output_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path

    # Step 2: Extract snapshots from the video at 5 seconds interval
    def extract_snapshots(self, video_path, interval=5):
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        snapshots = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Capture frame at every interval (e.g., 5 seconds)
            if int(cap.get(cv2.CAP_PROP_POS_MSEC)) // 1000 >= count * interval + 2:
                # Convert to grayscale (Step 3)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                snapshots.append(gray_frame)
                count += 1
        cap.release()
        return snapshots

    def create_animated_video(self, snapshots, video_duration, output_path='animated_video.mp4'):
        clips = []
        num_snapshots = len(snapshots)
        animation_duration = video_duration / num_snapshots  # Time each snapshot is shown in the final video

        # Mask for blackout gradation on left and right 10%
        h, w = snapshots[0].shape
        gradient_mask = np.ones((h, w), dtype=np.float32)
        gradient_width = int(w*0.1)
        for x in range(gradient_width):
            alpha = x/gradient_width
            gradient_mask[:,x] = alpha
        for x in range(w - gradient_width, w):
            alpha = (w - x)/gradient_width
            gradient_mask[:,x] = alpha
        
        for snapshot in snapshots:
            # Create slight horizontal translation animation (from right to left)
            frames = []
            num_frames = int(animation_duration * 60)  # Assuming 60 FPS
            for i in range(num_frames):
                translation_matrix = np.float32([[1.1, 0, -(i / num_frames) * w * 0.1], [0, 1.1, -h*0.05]])  # 10% horizontal translation
                translated_frame = cv2.warpAffine(snapshot, translation_matrix, (w, h))
                if i < 30:
                    percentage = i/30*0.5
                    translated_frame = (translated_frame*percentage)
                elif i >= num_frames - 31:
                    percentage = (1 - (i - num_frames + 31)/30)*0.5
                    translated_frame = (translated_frame*percentage)
                else:
                    translated_frame = (translated_frame*0.5)
                # Left and Right 10% gradient blackout
                masked_frame = (translated_frame*gradient_mask).astype(np.uint8)
                frames.append(masked_frame)

            # Convert frames to RGB for MoviePy compatibility
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in frames]

            # Create a clip from the frames
            clip = ImageSequenceClip(frames_rgb, fps=60)
            clips.append(clip)

        # Add dissolve transitions (crossfade) between clips
        return concatenate_videoclips(clips, method="compose") 

    def production_mode(self, youtube_url=None):
        
        # Read json file
        with open(self.MidiPath, 'r') as json_file:
            mididata = json.load(json_file)

        video_layers = []
        count = 0
        note_start_time = np.inf
        note_end_time = 0
        for i, track in enumerate(mididata['tracks']):
            pedal_value = 0
            pedal_direction = 0
            for c, sustain in enumerate(track['controlChanges']['64']):
                if sustain['number'] == 64:# pedal
                    new_direction = 0
                    if sustain['value'] > pedal_value:
                        new_direction = 1
                    elif sustain['value'] < pedal_value:
                        new_direction = -1
                    
                    pedal_value = sustain['value']
                    if new_direction != pedal_direction:
                        pedal_direction = new_direction
                        if new_direction == 1:
                            if True:
                                pedal_layer = self.create_note_clip(-100, sustain['time'], self.LongNoteThreshold/2)
                                video_layers.extend(pedal_layer)
                                count += 1
                            else:
                                break
            count = 0
            for j, note in enumerate(track['notes']):
                new_layers = self.create_note_clip(note['midi']-21, note['time'], note['duration'])
                if note_start_time > note['time']:
                    note_start_time = note['time']
                if note_end_time < note['time'] + note['duration']:
                    note_end_time = note['time'] + note['duration']
                video_layers.extend(new_layers)
                count += 1
        
        composite = CompositeVideoClip(video_layers, size=(1920,1080))

        # Process bga video if it is supplied
        if youtube_url != None:
            # Step 1: Download video
            print("Youtube url right here", youtube_url)
            youtube_video_path = self.download_youtube_video(youtube_url)

            # Step 2 & 3: Extract grayscale snapshots at 5 second intervals
            snapshots = self.extract_snapshots(youtube_video_path, interval=5)

            # Step 4: Create the animated video with dissolve transitions
            duration = note_end_time - note_start_time
            bga_clip = self.create_animated_video(snapshots, duration)
            bga_clip = bga_clip.resize(height=650).set_position(('center',174))
            bga_clip = bga_clip.set_start(note_start_time)
            final_mix = CompositeVideoClip([bga_clip, composite], size=(1920,1080))
            final_mix.write_videofile('synthesia_prod.mp4', threads=4)

            # Delete youtube bga file
            os.remove(youtube_video_path)
        else:
            composite.write_videofile('synthesia_prod.mp4', threads=4)


    def tutorial_mode(self, isCodeMajor, script_path):
        
        # FLIP HAND ? False for Track 1 LH and Track 2 RH, otherwise True
        FLIP_HAND = True

        # Read json file
        with open(self.MidiPath, 'r') as json_file:
            mididata = json.load(json_file)
        
        video_layers = []

        key_layers = []
        scripttime_offset = 2 # default time offset
        korean_english_switcher = 0
        script_font_size = 40
        if script_path != None:
            with open(script_path, 'r', encoding="utf-8") as file:
                text_duration = 0
                for line in file:
                    text_to_print = line.strip()
                    
                    text_y = 500
                    if korean_english_switcher == 0:
                        text_duration = len(text_to_print)*0.2 # 5 korean characters in 1 second
                        text_y = text_y - script_font_size - 10
                    textclip = TextClip(text_to_print, fontsize=script_font_size, color='white', font='MapoGoldenPier')
                    textclip = textclip.set_duration(text_duration)
                    textclip = textclip.set_start(scripttime_offset)

                    textclip = textclip.set_position(('center', text_y))
                    video_layers.append(textclip)
                    if korean_english_switcher == 1:
                        scripttime_offset += (text_duration + 0.5)
                    korean_english_switcher = (korean_english_switcher + 1)%2

        count = 0
        note_total_duration = 0
        note_starttime = scripttime_offset + 3
        hand_switcher = 0 # 1 for LH, 2 for RH

        # Mark finger numbers ui
        for i, track in enumerate(mididata['tracks']):
            if track['duration'] == 0:
                continue
            else:
                ms = musicsheet.MusicSheet(track['notes'], track['duration'], 20) # Mark finger numbers ui
                ms.run_ui()
        # Save finger numbers to json file
        with open(self.MidiPath, 'w') as f:
            json.dump(mididata, f)
        print("\nJSON file saved to", self.MidiPath)

        for i, track in enumerate(mididata['tracks']):
            if track['duration'] == 0:
                continue # Pass 
            else:
                if FLIP_HAND == True:
                    hand_switcher = (hand_switcher-1)%3
                else:
                    hand_switcher += 1

            pedal_value = 0
            pedal_direction = 0

            if '64' in track['controlChanges']:
                for c, sustain in enumerate(track['controlChanges']['64']):
                    if sustain['number'] == 64:# pedal
                        new_direction = 0
                        if sustain['value'] > pedal_value:
                            new_direction = 1
                        elif sustain['value'] < pedal_value:
                            new_direction = -1
                        
                        pedal_value = sustain['value']
                        if new_direction != pedal_direction:
                            pedal_direction = new_direction
                            if new_direction == 1:
                                if True:
                                    pedal_layer = self.create_note_clip(-100, note_starttime + sustain['time'], 0, True, isCodeMajor)
                                    video_layers.extend(pedal_layer)
                                    count += 1
                                else:
                                    break
            count = 0
            for j, note in enumerate(track['notes']):
                if True:
                    new_layers = None
                    if 'finger' in note:
                        new_layers = self.create_note_clip(note['midi']-21, note_starttime + note['time'], 0, True, isCodeMajor, hand_switcher, note['finger'])
                        key_layers.append(self.create_key(note['midi']-21, True, note_starttime + note['time'], 0.2, hand_switcher))
                        # key_layers.append(self.create_finger_under_key(note['midi']-21,  note_starttime + note['time'], note['finger']))
                    else:
                        new_layers = self.create_note_clip(note['midi']-21, note_starttime + note['time'], 0, True, isCodeMajor, hand_switcher)
                        key_layers.append(self.create_key(note['midi']-21, True, note_starttime + note['time'], 0.2, hand_switcher))
                    video_layers.extend(new_layers)
                    count += 1
                
                if (note['time'] + note['duration']) > note_total_duration:
                    note_total_duration = (note['time'] + note['duration'])
       
        # Set piano keys background
        video_layers.extend(self.create_full_keyboard_inactive(note_starttime + note_total_duration + 5))
        # Add active piano keys for tutorial
        video_layers.extend(key_layers)
        
        composite = CompositeVideoClip(video_layers, size=(1920,1080))
        composite.write_videofile('synthesia_tutorial.mp4', fps=60, threads=8)
    
    

def main():
    # Fixed constants    
    FrameRate = 60
    FallingHeight = 822 #pixels
    DefaultSpeed = FallingHeight/(140/FrameRate) #pixel per second
    RunTime = 1200

    # Customizable option. do not specify -x option in order to use this.
    FallingNotesSpeed = [{'t':0.0, 'v':DefaultSpeed*1.0}, {'t':RunTime, 'v':DefaultSpeed*1.0}]

    parser = argparse.ArgumentParser(description='synthesia')

    # Add command line options
    parser.add_argument('-t', action='store_true', help='Tutorial mode')
    parser.add_argument('-p', action='store_true', help='Production mode')
    parser.add_argument('-m', metavar='path', help='Path to the MIDI file')
    parser.add_argument('-b', type=int, help='bpm')
    parser.add_argument('-x', type=float, help='note speed x')
    parser.add_argument('-c', type=bool, help='code major (1) or minor (0) for tutorial')
    parser.add_argument('-s', metavar='path', help='Path to the tutorial script')
    parser.add_argument('-y', type=str, help='Optional for production: Link to youtube bga video')
    

    # Parse command line arguments
    args = parser.parse_args()
    print("Youtube argument" ,args.y)
    if not isinstance(args.y, str):
        print("Youtube argument is not a string!!")

    # Check if required options are provided
    if not (args.t or args.p):
        parser.error('Either -t (tutorial) or -p (production) must be specified.')

    if args.t and (not args.c):
        parser.error('when running tutorial mode, please specify if code is major or minor by -c 1 or -c 0')    

    if not args.m:
        parser.error('The -m option is required. Please specify the path to the midi file.')

    if args.p and (not args.b):
        parser.error("The -b option is required. Please specify bpm.")
    
    if args.t:
        args.b = 100 # Tutorial mode does not need bpm, as it does not create long note
    
    if not args.x:
        print("The -x option is not specified. Using the default note speed in the file.")
    else:
        FallingNotesSpeed = [{'t':0.0, 'v':DefaultSpeed*args.x}, {'t':RunTime, 'v':DefaultSpeed*args.x}]
    
    if args.t and (not args.s):
        print("The -s option is not specified. No script provided for this tutorial, skipping it.")

    # Initialize Synthesia
    print("arg b", args.b)
    synthesia_obj = Synthesia(args.b, FallingNotesSpeed, FallingHeight, DefaultSpeed, args.m)

    # Process command line options
    if args.t:
        print('Tutorial mode activated')
        synthesia_obj.tutorial_mode(args.c, args.s)

    elif args.p:
        print('Production mode activated')
        if args.y:
            synthesia_obj.production_mode(youtube_url=args.y)
        else:
            synthesia_obj.production_mode()

if __name__ == "__main__":
    main()