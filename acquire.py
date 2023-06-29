import cv2
import datetime
import time

durata_video = 15

numero_video = 6

# Inizializza la webcam
webcam = cv2.VideoCapture(1)
titolo = "Substitution_M_"

frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i in range(numero_video):
    start_time = datetime.datetime.now()

    writer = cv2.VideoWriter(titolo + str(i) + '.mp4', fourcc, 30.0, (frame_width, frame_height))
    while (datetime.datetime.now() - start_time).seconds < durata_video:
        ret, frame = webcam.read()

        if ret:
            secondi_restanti = durata_video - (datetime.datetime.now() - start_time).seconds

            testo = f"{titolo} - {secondi_restanti} secondi"
            writer.write(frame)
            cv2.putText(frame, testo, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Webcam', frame)



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Pausa di 3 secondi tra i video
    if i < numero_video - 1:
        time.sleep(3)

webcam.release()
writer.release()
cv2.destroyAllWindows()
