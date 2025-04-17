import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
from skimage.filters import gabor
import warnings
import tkinter as tk
from tkinter import messagebox, simpledialog

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Create directory for captures
save_dir = "iris_captures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize SQLite database
db_file = "iris_database.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS iris_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        account_number TEXT UNIQUE,
        balance REAL,
        pin TEXT,
        iris_features BLOB,
        timestamp TEXT,
        description TEXT
    )
''')
conn.commit()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar cascades.")
    exit()

# Tkinter GUI class for ATM-like interface
class ATMGUI:
    def __init__(self, root, account_number, balance, conn, cursor):
        self.root = root
        self.account_number = account_number
        self.balance = balance
        self.conn = conn
        self.cursor = cursor
        self.root.title("ATM Interface")
        self.root.geometry("600x500")
        self.root.configure(bg="#1C2526")  # Dark ATM-like background
        self.input_amount = ""

        # Screen area (mimics ATM display)
        self.screen_frame = tk.Frame(self.root, bg="#C1D8C3", bd=5, relief="sunken")
        self.screen_frame.pack(pady=20, padx=20, fill="x")
        self.screen_label = tk.Label(
            self.screen_frame, 
            text=f"Account: {account_number}\nBalance: ${balance:.2f}\nWelcome! Select an option.", 
            font=("Arial", 14, "bold"), 
            bg="#C1D8C3", 
            fg="#1C2526", 
            wraplength=500, 
            justify="center"
        )
        self.screen_label.pack(pady=10)

        # Function buttons frame
        self.function_frame = tk.Frame(self.root, bg="#1C2526")
        self.function_frame.pack(pady=10)

        # Function buttons (styled like ATM soft keys)
        button_style = {"font": ("Arial", 12, "bold"), "bg": "#4E6E5D", "fg": "white", "width": 15, "height": 2, "bd": 3, "relief": "raised"}
        tk.Button(self.function_frame, text="Balance Inquiry", command=self.balance_inquiry, **button_style).grid(row=0, column=0, padx=5)
        tk.Button(self.function_frame, text="Withdraw Cash", command=self.withdraw, **button_style).grid(row=0, column=1, padx=5)
        tk.Button(self.function_frame, text="Deposit Cash", command=self.deposit, **button_style).grid(row=1, column=0, padx=5)
        tk.Button(self.function_frame, text="Exit", command=self.exit, **button_style).grid(row=1, column=1, padx=5)

        # Numeric keypad frame
        self.keypad_frame = tk.Frame(self.root, bg="#1C2526")
        self.keypad_frame.pack(pady=10)

        # Numeric keypad buttons
        keypad_style = {"font": ("Arial", 12, "bold"), "bg": "#4E6E5D", "fg": "white", "width": 5, "height": 2, "bd": 3, "relief": "raised"}
        keypad_buttons = [
            ('1', 0, 0), ('2', 0, 1), ('3', 0, 2),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2),
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2),
            ('0', 3, 1), ('Clear', 3, 0), ('Enter', 3, 2)
        ]
        for (text, row, col) in keypad_buttons:
            if text in ['Clear', 'Enter']:
                cmd = self.clear_input if text == 'Clear' else self.enter_input
            else:
                cmd = lambda x=text: self.add_digit(x)
            tk.Button(self.keypad_frame, text=text, command=cmd, **keypad_style).grid(row=row, column=col, padx=5, pady=5)

        # Input display for keypad
        self.input_label = tk.Label(self.root, text="Amount: $0", font=("Arial", 12, "bold"), bg="#1C2526", fg="white")
        self.input_label.pack(pady=5)

    def add_digit(self, digit):
        self.input_amount += digit
        self.input_label.config(text=f"Amount: ${self.input_amount}")

    def clear_input(self):
        self.input_amount = ""
        self.input_label.config(text="Amount: $0")

    def enter_input(self):
        if not self.input_amount:
            messagebox.showerror("Error", "Please enter an amount.")
            return
        try:
            amount = float(self.input_amount)
            self.input_amount = ""
            self.input_label.config(text="Amount: $0")
            return amount
        except ValueError:
            messagebox.showerror("Error", "Invalid amount.")
            self.clear_input()

    def balance_inquiry(self):
        self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nYour balance is ${self.balance:.2f}")
        messagebox.showinfo("Balance", f"Your balance is ${self.balance:.2f}")

    def withdraw(self):
        amount = self.enter_input()
        if amount is None:
            return
        if amount <= 0:
            self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nInvalid amount.")
            messagebox.showerror("Error", "Invalid amount.")
        elif amount > self.balance:
            self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nInsufficient funds.")
            messagebox.showerror("Error", "Insufficient funds.")
        else:
            self.balance -= amount
            self.cursor.execute("UPDATE iris_data SET balance = ? WHERE account_number = ?", 
                              (self.balance, self.account_number))
            self.conn.commit()
            self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nWithdrawal successful.")
            messagebox.showinfo("Success", f"Withdrawal successful. New balance: ${self.balance:.2f}")

    def deposit(self):
        amount = self.enter_input()
        if amount is None:
            return
        if amount <= 0:
            self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nInvalid amount.")
            messagebox.showerror("Error", "Invalid amount.")
        else:
            self.balance += amount
            self.cursor.execute("UPDATE iris_data SET balance = ? WHERE account_number = ?", 
                              (self.balance, self.account_number))
            self.conn.commit()
            self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nDeposit successful.")
            messagebox.showinfo("Success", f"Deposit successful. New balance: ${self.balance:.2f}")

    def exit(self):
        self.screen_label.config(text=f"Account: {self.account_number}\nBalance: ${self.balance:.2f}\nThank you for using the ATM.")
        self.root.destroy()

def normalize_iris(image, center, radius):
    x, y = center
    iris_roi = image[max(0, y-radius):y+radius, max(0, x-radius):x+radius]
    if iris_roi.size == 0:
        return None
    iris_normalized = cv2.resize(iris_roi, (128, 64), interpolation=cv2.INTER_LINEAR)
    return iris_normalized

def extract_iris_features(image):
    if image is None:
        return None
    freq = 0.1
    theta = 0
    gabor_real, _ = gabor(image, frequency=freq, theta=theta)
    features = (gabor_real > np.mean(gabor_real)).astype(np.uint8).flatten()
    return features.tobytes()

def hamming_distance(features1, features2):
    f1 = np.frombuffer(features1, dtype=np.uint8)
    f2 = np.frombuffer(features2, dtype=np.uint8)
    return np.sum(f1 != f2) / len(f1)

def check_iris_in_database(iris_features, threshold=0.3):
    cursor.execute("SELECT id, user_id, account_number, balance, pin, iris_features FROM iris_data")
    for row in cursor.fetchall():
        db_id, user_id, account_number, balance, pin, stored_features = row
        distance = hamming_distance(iris_features, stored_features)
        if distance < threshold:
            return True, user_id, account_number, balance, pin, stored_features
    return False, None, None, None, None, None

def save_iris_to_database(user_id, account_number, balance, pin, iris_features, description):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cursor.execute(
        "INSERT INTO iris_data (user_id, account_number, balance, pin, iris_features, timestamp, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, account_number, balance, pin, iris_features, timestamp, description)
    )
    conn.commit()

def overlay_text_on_image(image, text):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    position = (10, 20)
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def perform_second_iris_scan(stored_features, root, max_attempts=3, threshold=0.3):
    attempts = 0
    while attempts < max_attempts:
        messagebox.showinfo("Second Scan", f"Attempt {attempts + 1}/{max_attempts}: Scan your iris again to confirm.")
        iris_detected = False
        iris_normalized = None

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Could not read frame.")
                return False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                    eye_roi = cv2.GaussianBlur(eye_roi, (5, 5), 0)

                    circles = cv2.HoughCircles(
                        eye_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                        param1=50, param2=30, minRadius=10, maxRadius=50
                    )

                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        for (cx, cy, r) in circles:
                            cv2.circle(roi_color, (cx, cy), r, (0, 0, 255), 2)
                            iris_normalized = normalize_iris(eye_roi, (cx, cy), r)
                            if iris_normalized is not None:
                                iris_detected = True
                                cv2.imshow('Second Iris Scan', iris_normalized)

            cv2.putText(frame, "Scan iris again to confirm (Press 'c' to capture)", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Webcam - Second Iris Scan', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and iris_detected:
                second_iris_features = extract_iris_features(iris_normalized)
                if second_iris_features:
                    distance = hamming_distance(second_iris_features, stored_features)
                    if distance < threshold:
                        messagebox.showinfo("Success", "Second iris scan verified.")
                        cv2.destroyWindow('Second Iris Scan')
                        return True
                    else:
                        messagebox.showerror("Error", "Second iris scan does not match.")
                        attempts += 1
                        break
            elif key == ord('q'):
                messagebox.showinfo("Cancelled", "Second scan cancelled.")
                cv2.destroyWindow('Second Iris Scan')
                return False

        if attempts < max_attempts:
            messagebox.showinfo("Retry", "Please try again.")

    messagebox.showerror("Error", f"Failed to verify iris after {max_attempts} attempts. Access denied.")
    cv2.destroyWindow('Second Iris Scan')
    return False

def register_user(root, iris_features):
    user_id = simpledialog.askstring("Register", "Enter user ID (e.g., name):", parent=root)
    if user_id is None:
        return False
    user_id = user_id.strip() or "unknown"

    account_number = simpledialog.askstring("Register", "Enter account number:", parent=root)
    if account_number is None:
        return False
    account_number = account_number.strip()

    balance = simpledialog.askfloat("Register", "Enter initial balance:", parent=root)
    if balance is None:
        return False

    pin = simpledialog.askstring("Register", "Set PIN (optional, for backup):", parent=root)
    if pin is None:
        pin = "none"
    else:
        pin = pin.strip() or "none"

    description = simpledialog.askstring("Register", "Enter description (e.g., left eye):", parent=root)
    if description is None:
        description = "No description"
    else:
        description = description.strip() or "No description"

    # Check if account number is unique
    cursor.execute("SELECT account_number FROM iris_data WHERE account_number = ?", (account_number,))
    if cursor.fetchone():
        messagebox.showerror("Error", "Account number already exists.")
        return False

    save_iris_to_database(user_id, account_number, balance, pin, iris_features, description)
    
    # Save iris image with description
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iris_filename = os.path.join(save_dir, f'iris_{timestamp}.png')
    iris_with_text = overlay_text_on_image(iris_normalized, description)
    cv2.imwrite(iris_filename, iris_with_text)
    messagebox.showinfo("Success", f"Registered user and saved iris image: {iris_filename}")
    return True

def main_loop():
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window until needed

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        iris_detected = False
        global iris_normalized
        iris_normalized = None

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_color = roi_color[ey:ey+eh, ex:ex+ew]
                eye_roi = cv2.GaussianBlur(eye_roi, (5, 5), 0)

                circles = cv2.HoughCircles(
                    eye_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=50, param2=30, minRadius=10, maxRadius=50
                )

                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    for (cx, cy, r) in circles:
                        cv2.circle(eye_color, (cx, cy), r, (0, 0, 255), 2)
                        iris_normalized = normalize_iris(eye_roi, (cx, cy), r)
                        if iris_normalized is not None:
                            iris_detected = True
                            cv2.imshow('Normalized Iris', iris_normalized)

        cv2.putText(frame, "Press 's' to scan iris", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Webcam - Iris Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and iris_detected:
            iris_features = extract_iris_features(iris_normalized)
            if iris_features:
                is_present, user_id, account_number, balance, pin, stored_features = check_iris_in_database(iris_features)
                if is_present:
                    messagebox.showinfo("Success", f"Iris authenticated for user: {user_id}")
                    if perform_second_iris_scan(stored_features, root):
                        # Show ATM GUI
                        atm_root = tk.Toplevel(root)
                        ATMGUI(atm_root, account_number, balance, conn, cursor)
                        atm_root.wait_window()  # Wait for ATM GUI to close
                    else:
                        messagebox.showerror("Error", "Second iris verification failed. Access denied.")
                else:
                    messagebox.showinfo("Register", "Iris not found in database. Please register.")
                    if register_user(root, iris_features):
                        messagebox.showinfo("Success", "Registration complete. Scan iris again to access ATM.")
                    else:
                        messagebox.showerror("Error", "Registration cancelled or failed.")

        if key == ord('q'):
            break

    # Release resources
    cap.release()
    conn.close()
    cv2.destroyAllWindows()
    root.destroy()

if __name__ == "__main__":
    main_loop()
