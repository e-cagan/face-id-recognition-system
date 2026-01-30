"""Face ID Recognition System - GUI Application."""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2

from modules import FaceDetector, FaceRecognizer, UserRegistration, DataManager


class FaceIDApp:
    """Main GUI application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face ID Recognition System")
        self.root.resizable(False, False)

        # Initialize modules
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.data_manager = DataManager()
        self.registration = UserRegistration(self.detector, self.recognizer, self.data_manager)

        self.data_manager.connect()

        # State
        self.camera_running = False
        self.current_mode = None  # 'register' or 'verify'

        self._setup_ui()

    def _setup_ui(self):
        """Create UI components."""
        # Camera frame - use placeholder image for correct sizing
        self.camera_label = tk.Label(self.root, bg="black")
        self.camera_label.pack(pady=10)
        
        # Set initial size with placeholder
        placeholder = Image.new('RGB', (640, 480), 'black')
        self.placeholder_img = ImageTk.PhotoImage(placeholder)
        self.camera_label.configure(image=self.placeholder_img)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Input frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="User ID:").grid(row=0, column=0, padx=5)
        self.user_id_entry = tk.Entry(input_frame, width=20)
        self.user_id_entry.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Name:").grid(row=0, column=2, padx=5)
        self.name_entry = tk.Entry(input_frame, width=20)
        self.name_entry.grid(row=0, column=3, padx=5)

        # Button frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.register_btn = tk.Button(btn_frame, text="Register", width=12, command=self.start_register)
        self.register_btn.grid(row=0, column=0, padx=5)

        self.verify_btn = tk.Button(btn_frame, text="Verify", width=12, command=self.start_verify)
        self.verify_btn.grid(row=0, column=1, padx=5)

        self.capture_btn = tk.Button(btn_frame, text="Capture", width=12, command=self.capture, state=tk.DISABLED)
        self.capture_btn.grid(row=0, column=2, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop", width=12, command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=3, padx=5)

        # User list frame
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=10, fill=tk.X, padx=20)

        tk.Label(list_frame, text="Registered Users:").pack(anchor=tk.W)

        self.user_listbox = tk.Listbox(list_frame, height=5)
        self.user_listbox.pack(fill=tk.X, pady=5)

        list_btn_frame = tk.Frame(list_frame)
        list_btn_frame.pack(fill=tk.X)

        tk.Button(list_btn_frame, text="Refresh", command=self.refresh_user_list).pack(side=tk.LEFT)
        tk.Button(list_btn_frame, text="Delete Selected", command=self.delete_selected_user).pack(side=tk.LEFT, padx=5)

        self.refresh_user_list()

    def start_register(self):
        """Start camera for registration."""
        user_id = self.user_id_entry.get().strip()
        name = self.name_entry.get().strip()

        if not user_id or not name:
            messagebox.showwarning("Input Error", "Please enter User ID and Name.")
            return

        if self.data_manager.user_exists(user_id):
            messagebox.showwarning("Error", "User ID already exists.")
            return

        self.current_mode = 'register'
        self._start_camera()
        self.status_var.set("Position your face and click Capture")

    def start_verify(self):
        """Start camera for verification."""
        self.current_mode = 'verify'
        self._start_camera()
        self.status_var.set("Position your face and click Capture")

    def _start_camera(self):
        """Initialize and start camera feed."""
        if not self.detector.start_camera():
            messagebox.showerror("Error", "Failed to open camera.")
            return

        self.camera_running = True
        self.capture_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.register_btn.config(state=tk.DISABLED)
        self.verify_btn.config(state=tk.DISABLED)

        self._update_frame()

    def _update_frame(self):
        """Update camera frame in UI."""
        if not self.camera_running:
            return

        frame = self.detector.get_frame()
        if frame is not None:
            # Detect face and draw bbox
            face_data = self.detector.detect_face(frame)
            if face_data:
                frame = self.detector.draw_bbox(frame, face_data['region'])

            # Convert to tkinter format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        self.root.after(30, self._update_frame)

    def capture(self):
        """Capture current frame and process."""
        frame = self.detector.get_frame()
        if frame is None:
            messagebox.showerror("Error", "Failed to capture frame.")
            return

        if self.current_mode == 'register':
            self._process_registration(frame)
        elif self.current_mode == 'verify':
            self._process_verification(frame)

    def _process_registration(self, frame):
        """Process registration with captured frame."""
        user_id = self.user_id_entry.get().strip()
        name = self.name_entry.get().strip()

        success, message = self.registration.register_from_image(user_id, name, frame)

        if success:
            messagebox.showinfo("Success", message)
            self.user_id_entry.delete(0, tk.END)
            self.name_entry.delete(0, tk.END)
            self.refresh_user_list()
            self.stop_camera()
        else:
            messagebox.showerror("Error", message)

    def _process_verification(self, frame):
        """Process verification with captured frame."""
        # Extract embedding from frame
        embedding = self.recognizer.extract_embedding(frame)
        if embedding is None:
            messagebox.showerror("Error", "No face detected.")
            return

        # Get all stored embeddings
        stored = self.data_manager.get_all_embeddings()
        if not stored:
            messagebox.showinfo("Info", "No registered users.")
            return

        # Find match
        match = self.recognizer.find_match(embedding, stored)
        if match:
            user_id, distance = match
            user = self.data_manager.get_user(user_id)
            self.status_var.set(f"Verified: {user['name']} ({user_id})")
            messagebox.showinfo("Verified", f"Welcome, {user['name']}!\nConfidence: {1 - distance:.2%}")
        else:
            self.status_var.set("Verification failed - Unknown face")
            messagebox.showwarning("Failed", "Face not recognized.")

    def stop_camera(self):
        """Stop camera feed."""
        self.camera_running = False
        self.detector.stop_camera()
        self.current_mode = None

        self.capture_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.register_btn.config(state=tk.NORMAL)
        self.verify_btn.config(state=tk.NORMAL)

        # Restore placeholder
        self.camera_label.configure(image=self.placeholder_img)
        self.status_var.set("Ready")

    def refresh_user_list(self):
        """Refresh the user listbox."""
        self.user_listbox.delete(0, tk.END)
        embeddings = self.data_manager.get_all_embeddings()
        for user_id, _ in embeddings:
            user = self.data_manager.get_user(user_id)
            self.user_listbox.insert(tk.END, f"{user_id} - {user['name']}")

    def delete_selected_user(self):
        """Delete selected user from list."""
        selection = self.user_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a user.")
            return

        item = self.user_listbox.get(selection[0])
        user_id = item.split(" - ")[0]

        if messagebox.askyesno("Confirm", f"Delete user '{user_id}'?"):
            if self.data_manager.delete_user(user_id):
                self.refresh_user_list()
                messagebox.showinfo("Success", "User deleted.")

    def on_closing(self):
        """Cleanup on window close."""
        self.stop_camera()
        self.data_manager.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceIDApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()