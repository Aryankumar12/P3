from flask import Flask, render_template, Response, request, redirect, url_for

import cv2
import pickle
import mediapipe as mp
import numpy as np

app = Flask(__name__)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'Z', 25: 'L', 26: '1'
}

@app.route('/')
def index():
    return render_template('choose_category.html')

@app.route('/choose_category', methods=['GET', 'POST'])
def choose_category():
    if request.method == 'POST':
        category = request.form['category']
        if category == 'numbers':
            return render_template('numbers.html')
        elif category == 'alphabets':
            return render_template('alphabets.html', labels_dict=labels_dict)
    return render_template('choose_category.html')

@app.route('/subpage', methods=['GET', 'POST'])
def subpage():
    if request.method == 'POST':
        selected_alphabet = request.form['selected_alphabet']
        return render_template('subpage.html', selected_alphabet=selected_alphabet)
    else:
        return redirect(url_for('choose_category'))

def generate_frames(selected_alphabet):
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_alpha = labels_dict[int(prediction[0])]

            print(predicted_alpha)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_alpha, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()




@app.route('/video_feed/<selected_alphabet>')
def video_feed(selected_alphabet):
    return Response(generate_frames(selected_alphabet), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


