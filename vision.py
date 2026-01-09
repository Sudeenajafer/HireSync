# phase2_gaze.py (save this)
import cv2
import mediapipe as mp
import numpy as np

print("🚀 HireSync Phase 2: MediaPipe Gaze ✅")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    h, w, _ = frame.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # LEFT eye center (468=inner, 473=outer)
            left_inner = face_landmarks.landmark[468]
            left_outer = face_landmarks.landmark[473]
            left_eye_center = (left_inner.x + left_outer.x) / 2 * w
            
            # RIGHT eye center (227=inner, 230=outer)
            right_inner = face_landmarks.landmark[227]
            right_outer = face_landmarks.landmark[230]
            right_eye_center = (right_inner.x + right_outer.x) / 2 * w
            
            # Gaze ratio (left-right eye distance normalized)
            gaze_ratio = abs(left_eye_center - right_eye_center) / w
            attention = 1.0 - gaze_ratio  # 1.0 = looking center
            
            cv2.circle(frame, (int(left_eye_center), int(left_inner.y * h)), 3, (0,255,0), -1)
            cv2.circle(frame, (int(right_eye_center), int(right_inner.y * h)), 3, (0,255,0), -1)
            
            cv2.putText(frame, f"Gaze: {gaze_ratio:.2f} | Attention: {attention:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            mp_drawing.draw_landmarks(frame, face_landmarks, 
                                    mp_face_mesh.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255)))
    
    cv2.imshow('HireSync Phase 2 Gaze', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👁️ Phase 2 Gaze Complete!")
