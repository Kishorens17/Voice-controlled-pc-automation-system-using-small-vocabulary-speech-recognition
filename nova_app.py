"""
Nova Premium Dashboard
Modern Voice Assistant GUI using CustomTkinter.
"""
import customtkinter as ctk
import threading
import time
import queue
import os
from voice_service import VoiceService
from typing import Optional, Tuple, Any

# ======================== Configuration ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "speech_crnn.keras")
CMD_THRESHOLD = 0.50
WAKE_THRESHOLD = 0.32 # Increased sensitivity for Nova


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class NovaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Nova Voice Assistant")
        self.geometry("500x450")
        self.resizable(False, False)

        # System initialization
        self.service = VoiceService(MODEL_PATH)
        self.running = False
        self.msg_queue = queue.Queue()
        self.history = []

        # UI Initialization
        self.setup_ui()

        # Popup handle
        self.popup: Optional[ctk.CTkToplevel] = None
        
        # Start command processing
        self.after(100, self.process_queue)

    def setup_ui(self):
        # Sidebar/Top bar style
        self.grid_columnconfigure(0, weight=1)
        
        # Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, height=80, fg_color="#1a1a1a")
        self.header_frame.grid(row=0, column=0, sticky="nsew")
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="NOVA AI", 
                                       font=ctk.CTkFont(size=24, weight="bold"),
                                       text_color="#00D4FF")
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")

        # Main Content Frame
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Status Ring / Indicator (Simulated with a label for now)
        self.status_indicator = ctk.CTkLabel(self.main_frame, text="●", font=("Helvetica", 48), text_color="#555555")
        self.status_indicator.grid(row=0, column=0, pady=(0, 10))

        self.status_text = ctk.CTkLabel(self.main_frame, text="SYSTEM IDLE", 
                                      font=ctk.CTkFont(size=14, weight="bold"),
                                      text_color="#888888")
        self.status_text.grid(row=1, column=0, pady=(0, 20))

        # Power Button
        self.power_btn = ctk.CTkButton(self.main_frame, text="START SERVICE", 
                                      command=self.toggle_service,
                                      height=45, corner_radius=10,
                                      font=ctk.CTkFont(size=14, weight="bold"),
                                      fg_color="#333333", hover_color="#444444")
        self.power_btn.grid(row=2, column=0, padx=50, sticky="ew")

        # History Box
        self.history_frame = ctk.CTkFrame(self.main_frame, fg_color="#151515", corner_radius=10)
        self.history_frame.grid(row=3, column=0, pady=(30, 0), sticky="nsew")
        self.history_frame.grid_columnconfigure(0, weight=1)

        self.history_label = ctk.CTkLabel(self.history_frame, text="RECENT ACTIVITY", 
                                        font=ctk.CTkFont(size=11, weight="bold"),
                                        text_color="#555555")
        self.history_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky="w")

        self.history_list = ctk.CTkLabel(self.history_frame, text="Waiting for commands...", 
                                       font=ctk.CTkFont(size=12),
                                       text_color="#aaaaaa", justify="left")
        self.history_list.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    def toggle_service(self):
        if not self.running:
            self.running = True
            self.power_btn.configure(text="STOP SERVICE", fg_color="#E63946", hover_color="#D62828")
            self.status_indicator.configure(text_color="#00FF88")
            self.status_text.configure(text="LISTENING FOR 'NOVA'", text_color="#00FF88")
            self.log_activity("System online")
            
            self.thread = threading.Thread(target=self.voice_listening_loop, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self.power_btn.configure(text="START SERVICE", fg_color="#333333", hover_color="#444444")
            self.status_indicator.configure(text_color="#555555")
            self.status_text.configure(text="SYSTEM IDLE", text_color="#888888")
            self.log_activity("System offline")

    def log_activity(self, text):
        t = time.strftime("%H:%M")
        self.history.append(f"[{t}] {text}")
        if len(self.history) > 3:
            self.history.pop(0)
        self.history_list.configure(text="\n".join(self.history))

    def voice_listening_loop(self):
        while self.running:
            try:
                audio = self.service.record_chunk()
                pred, conf = self.service.predict(audio)
                
                if pred == "Nova" and conf >= WAKE_THRESHOLD:
                    self.msg_queue.put(("SET_STATUS", ("HEARING...", "#FFB703")))
                    self.msg_queue.put(("SHOW_POPUP", ""))
                    
                    # Command window
                    time.sleep(1.0)
                    cmd_audio = self.service.record_chunk()
                    cmd_pred, cmd_conf = self.service.predict(cmd_audio)

                    if cmd_conf >= CMD_THRESHOLD and cmd_pred not in ["Background", "Nova"]:
                        self.msg_queue.put(("LOG", f"Command: {cmd_pred}"))
                        self.msg_queue.put(("EXECUTE", cmd_pred))
                    else:
                        self.msg_queue.put(("LOG", "No command heard"))
                        self.msg_queue.put(("CLOSE_POPUP", ""))
                    
                    self.msg_queue.put(("SET_STATUS", ("LISTENING FOR 'NOVA'", "#00FF88")))
                else:
                    if pred != "Background":
                        print(f"Ignored: {pred} ({conf:.1%})")
            except Exception as e:
                print(f"Loop error: {e}")
                time.sleep(1)

    def process_queue(self):
        try:
            while not self.msg_queue.empty():
                msg_type, data = self.msg_queue.get_nowait()
                if msg_type == "SHOW_POPUP":
                    self.show_popup_window()
                elif msg_type == "EXECUTE":
                    self.service.execute_action(data)
                    self.close_popup_window()
                elif msg_type == "CLOSE_POPUP":
                    self.close_popup_window()
                elif msg_type == "SET_STATUS":
                    txt, color = data
                    self.status_text.configure(text=txt, text_color=color)
                    self.status_indicator.configure(text_color=color)
                elif msg_type == "LOG":
                    self.log_activity(data)
        finally:
            self.after(100, self.process_queue)

    def show_popup_window(self):
        if self.popup: return
        
        self.popup = ctk.CTkToplevel(self)
        self.popup.title("Assistant")
        
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.popup.geometry(f"320x160+{sw-350}+{sh-220}")
        self.popup.overrideredirect(True)
        self.popup.attributes("-topmost", True)
        self.popup.configure(fg_color="#1a1a1a")

        container = ctk.CTkFrame(self.popup, corner_radius=15, fg_color="#222222", border_width=2, border_color="#00D4FF")
        container.pack(fill="both", expand=True, padx=5, pady=5)

        msg = self.service.get_random_greeting()
        lbl = ctk.CTkLabel(container, text=msg, font=ctk.CTkFont(size=14, weight="bold"),
                          text_color="#FFFFFF", wraplength=250)
        lbl.pack(pady=(40, 10))

        hint = ctk.CTkLabel(container, text="Listening for command...", font=ctk.CTkFont(size=11),
                           text_color="#00D4FF")
        hint.pack()

    def close_popup_window(self):
        if self.popup:
            self.popup.destroy()
            self.popup = None

if __name__ == "__main__":
    app = NovaApp()
    app.mainloop()
