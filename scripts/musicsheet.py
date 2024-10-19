import matplotlib.pyplot as plt
import json

class MusicSheet:
    def __init__(self, notes, total_duration, screentime_unit):
        self.notes = notes
        self.total_duration = total_duration
        self.current_time = 0
        self.screentime_unit = screentime_unit
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        self.fig = fig
        self.ax = ax
        fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.current_notes = []
        self.selected_note_index = -1

    def on_press(self, event):
        if event.key == 'pageup':
            self.go_previous()
        elif event.key == 'pagedown':
            self.go_next()
        elif event.key == 'escape':
            plt.close()
        elif event.key in ['left', 'right']:
            self.change_highlight(event.key)
        elif event.key.isdigit():
            self.mark_finger_number(event.key)

    def mark_finger_number(self, finger):
        if self.selected_note_index > -1:
            self.current_notes[self.selected_note_index]['finger'] = finger
            self.update_notes()

    def change_highlight(self, direction):
        if direction == 'left':
            self.selected_note_index = max(0, self.selected_note_index - 1)
        elif direction == 'right':
            self.selected_note_index = min(len(self.current_notes)-1, self.selected_note_index + 1)
        self.update_notes()

    def update_notes(self):        
        self.ax.clear()
        self.draw_notes()
        plt.draw()
        plt.pause(0.001)

    def draw_notes(self):
        start_time = self.current_time
        end_time = start_time + self.screentime_unit
        
        if self.selected_note_index == -1:
            # New page
            for note in self.notes:
                if start_time <= note["time"] <= end_time:
                    self.current_notes.append(note)
                    self.ax.plot([note["time"], note["time"] + note["duration"]],
                            [note["midi"], note["midi"]],
                            color='black')
                    note_text = note["name"]
                    if "finger" in note:
                        note_text += (":" + note['finger'])
                    self.ax.text(note["time"], note["midi"], note_text,
                            verticalalignment='bottom', fontsize=8)

        else:
            # Navigating in current page
            for i, note in enumerate(self.current_notes):
                note_color = 'black'
                if i == self.selected_note_index:
                    note_color = 'red'
                note_text = note["name"]
                if "finger" in note:
                    note_text += (":" + note['finger'])
                self.ax.plot([note["time"], note["time"] + note["duration"]],
                        [note["midi"], note["midi"]],
                        color=note_color)
                self.ax.text(note["time"], note["midi"], note_text,
                        verticalalignment='bottom', fontsize=8, color=note_color)
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("MIDI Note")
        self.ax.set_title("left/right:select note pgup/dwn:change page esc:exit")
        self.ax.grid(True)
        self.ax.set_ylim(0, 127)
        self.ax.set_xlim(start_time, end_time)

    def go_previous(self):
        self.current_time = max(0, self.current_time - self.screentime_unit)
        self.selected_note_index = -1
        self.current_notes = []
        self.update_notes()
    
    def go_next(self):
        self.current_time = min(self.total_duration, self.current_time + self.screentime_unit)
        self.selected_note_index = -1
        self.current_notes = []
        self.update_notes()
    
    def run_ui(self):
        self.draw_notes()
        plt.show()


# Example usage
# filename = 'legendofmoonlight_tutorial.json'
# midijson = None
# with open(filename, 'r') as f:
#     midijson = json.load(f)
# track = midijson['tracks'][1]
# total_duration = track['duration']
# notes = track['notes']

# ms = MusicSheet(notes, total_duration, 30)
# ms.run_ui()