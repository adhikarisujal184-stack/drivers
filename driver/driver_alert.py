import cv2
import time
import pygame
import numpy as np

# ---------------- SOUND FUNCTION ----------------
def init_sound():
    pygame.mixer.init(frequency=44100, size=-16, channels=2)

def generate_alarm_sound():
    sample_rate = 44100
    duration = 1
    frequency = 1000

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)

    audio = (tone * 32767).astype(np.int16)

    # convert mono to stereo
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)

def play_alarm(sound, is_playing):
    if not is_playing:
        sound.play(-1)  # loop
        return True
    return is_playing

def stop_alarm(sound, is_playing):
    if is_playing:
        sound.stop()
        return False
    return is_playing

# ---------------- DETECTION FUNCTION ----------------
def detect_eyes(gray, face_cascade, eye_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            eyes_detected = True

    return eyes_detected

# ---------------- MAIN FUNCTION ----------------
def main():
    init_sound()
    alarm_sound = generate_alarm_sound()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                        'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    eye_closed_start = None
    ALARM_DELAY = 3
    alarm_playing = False

    system_active = True   # 🔥 NEW: System ON/OFF

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- DISPLAY SYSTEM STATUS --------
        status_text = "SYSTEM ON" if system_active else "SYSTEM OFF"
        color = (0, 255, 0) if system_active else (0, 0, 255)

        cv2.putText(frame, status_text, (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if system_active:
            eyes_detected = detect_eyes(gray, face_cascade, eye_cascade)

            # -------- LOGIC --------
            if not eyes_detected:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    elapsed = time.time() - eye_closed_start

                    cv2.putText(frame, f"Closed: {int(elapsed)}s", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if elapsed >= ALARM_DELAY:
                        cv2.putText(frame, "DROWSINESS ALERT!", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        alarm_playing = play_alarm(alarm_sound, alarm_playing)
            else:
                eye_closed_start = None
                cv2.putText(frame, "Eyes Open", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                alarm_playing = stop_alarm(alarm_sound, alarm_playing)

        else:
            # 🔥 If system OFF → stop everything
            eye_closed_start = None
            alarm_playing = stop_alarm(alarm_sound, alarm_playing)

        cv2.imshow("Driver Alert System", frame)

        key = cv2.waitKey(1) & 0xFF

        # 🔥 TOGGLE SYSTEM (PRESS 'S')
        if key == ord('s'):
            system_active = not system_active

        # 🔥 EXIT (ESC)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()