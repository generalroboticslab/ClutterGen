import threading
import tkinter as tk
from tkinter import ttk, messagebox
import speech_recognition as sr
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import time
import os

class FeedbackGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Feedback GUI")

        # Bind a function to handle script termination
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.create_layers()
        
        self.root.mainloop()

    def create_layers(self):
        self.create_layer1()
        self.create_layer2()
        self.create_layer3()
        self.create_layer4()
        self.create_layer5()

    def create_layer1(self):
        self.layer1_frame = ttk.LabelFrame(self.root, text="Feedback Buttons")
        self.layer1_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.create_feedback_button("Good", 0, 0)
        self.create_feedback_button("Bad", 0, 1)
        self.create_feedback_button("Neutral", 0, 2)

    def create_feedback_button(self, label, row, column):
        button = ttk.Button(self.layer1_frame, text=label, command=lambda: self.record_feedback(label))
        button.grid(row=row, column=column, padx=5, pady=5)

    def create_layer2(self):
        self.layer2_frame = ttk.LabelFrame(self.root, text="Text Input")
        self.layer2_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.text_entry = tk.Text(self.layer2_frame, wrap="word", height=5, width=50)
        self.text_entry.grid(row=0, column=0, padx=10, pady=5)

        self.send_text_button = ttk.Button(self.layer2_frame, text="Send Text", command=self.send_text)
        self.send_text_button.grid(row=1, column=0, padx=10, pady=5)

        self.text_entry.bind("<Return>", self.handle_enter_key)
        self.text_entry.bind("<Shift-Return>", self.insert_newline)

    def create_layer3(self):
        self.layer3_frame = ttk.LabelFrame(self.root, text="Keyboard Input")
        self.layer3_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.keyboard_input_var = tk.StringVar()
        self.keyboard_checkbutton = ttk.Checkbutton(self.layer3_frame, text="Enable Keyboard Input", variable=self.keyboard_input_var)
        self.keyboard_checkbutton.grid(row=0, column=0, padx=10, pady=5)

    def create_layer4(self):
        self.layer4_frame = ttk.LabelFrame(self.root, text="Voice Input")
        self.layer4_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.voice_input_button = ttk.Button(self.layer4_frame, text="Start Voice Input", command=self.start_voice_input)
        self.voice_input_button.grid(row=0, column=0, padx=10, pady=5)

        self.voice_output_text = tk.Text(self.layer4_frame, wrap="word", height=5, width=50)
        self.voice_output_text.grid(row=1, column=0, padx=10, pady=5)

    def create_layer5(self):
        self.audio_layer_frame = ttk.LabelFrame(self.root, text="Audio Input with Waveform Plot")
        self.audio_layer_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.figure, self.ax = plt.subplots(figsize=(2, 3))
        self.line, = self.ax.plot([], [])
        self.ax.set_ylim(-1, 1)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.audio_layer_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.timer_label = ttk.Label(self.audio_layer_frame, text="Timer: 00:00")
        self.timer_label.pack(padx=10, pady=5)

        self.animation = FuncAnimation(self.figure, self.update_plot, blit=True, interval=100)

        record_button_frame = ttk.Frame(self.audio_layer_frame)
        record_button_frame.pack(padx=10, pady=5)

        self.record_button = ttk.Button(record_button_frame, text="Start Recording", command=self.start_recording)
        self.record_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(record_button_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(side="left", padx=5)

        self.recording_start_time = None
        self.recording_duration = 5

    def start_recording(self):
        self.stop_recording()  # Stop previous recording if any
        self.recording_start_time = time.time()
        self.recording = sd.rec(int(self.recording_duration * 44100), samplerate=44100, channels=1)
        self.animation.event_source.start()
        self.after_callback = self.root.after(int(self.recording_duration * 1000), self.stop_recording)

    def stop_recording(self):
        if hasattr(self, 'recording'):
            sd.stop()
            self.animation.event_source.stop()
            self.recording_start_time = None
            if self.after_callback:
                self.root.after_cancel(self.after_callback)
                self.after_callback = None
            
    def update_plot(self, frame):
        if hasattr(self, 'recording'):
            self.line.set_data(self.get_time_axis(), self.recording)
            self.ax.set_xlim(0, self.recording_duration)
            self.update_timer_label()
            self.canvas.draw()

        return self.line,

    def get_time_axis(self):
        if hasattr(self, 'recording'):
            time_axis = [(t / 44100) for t in range(len(self.recording))]
            return time_axis
        return []

    def update_timer_label(self):
        if self.recording_start_time is not None:
            elapsed_time = time.time() - self.recording_start_time
            minutes = int(elapsed_time / 60)
            seconds = int(elapsed_time % 60)
            self.timer_label.config(text=f"Timer: {minutes:02}:{seconds:02}")

    def record_feedback(self, feedback_type):
        messagebox.showinfo("Feedback Recorded", f"Feedback recorded: {feedback_type}")

    def handle_enter_key(self, event):
        self.send_text()
        return "break"

    def insert_newline(self, event):
        self.text_entry.insert("insert", "\n")
        return "break"

    def send_text(self):
        text = self.text_entry.get("1.0", "end-1c")
        if text.strip():
            self.text_entry.delete("1.0", "end")
            messagebox.showinfo("Text Sent", f"Text sent: {text}")

    def start_voice_input(self):
        self.voice_output_text.delete("1.0", "end")
        self.voice_output_text.insert("1.0", "Listening... (Speak now)")
        self.voice_output_text.update()

        threading.Thread(target=self.process_voice_input).start()

    def process_voice_input(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=1)
                text = recognizer.recognize_google(audio)
                self.voice_output_text.delete("1.0", "end")
                self.voice_output_text.insert("1.0", f"Voice Input: {text}")
            except sr.WaitTimeoutError:
                self.voice_output_text.delete("1.0", "end")
                self.voice_output_text.insert("1.0", "Voice input timeout")
            except sr.UnknownValueError:
                self.voice_output_text.delete("1.0", "end")
                self.voice_output_text.insert("1.0", "Could not understand audio")
            except sr.RequestError as e:
                self.voice_output_text.delete("1.0", "end")
                self.voice_output_text.insert("1.0", f"Error: {str(e)}")

    def on_closing(self):
        self.stop_recording()
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    gui = FeedbackGUI()
