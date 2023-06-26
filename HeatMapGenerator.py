import copy
import cv2
import numpy as np


class HeatMapGenerator:

    def __init__(self):
        self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        self.first_frame = None
        self.accum_image = None
        self.result_overlay = None

    def get_result_overlay(self):
        return self.result_overlay

    def clean(self):
        print("pulisco immagine")
        self.result_overlay = None

    def set_result_overlay(self, value):
        self.result_overlay = value

    def generate_heatmap(self, frame, first_iteration_indicator):

        if first_iteration_indicator == 1:
            self.first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            self.accum_image = np.zeros((height, width), np.uint8)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = self.fgbg.apply(gray)  # remove the background

            # Applica una soglia binaria mantenendo solo i pixel al di sopra della soglia e impostando il risultato su
            # maxValue. Se vuoi che il movimento venga rilevato di più, aumenta il valore di maxValue.
            # Per rilevare la minor quantità di movimento nel tempo, imposta maxValue = 1.
            thresh = 2
            maxValue = 50
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)

            # Aggiunge all'immagine di accumulo
            self.accum_image = cv2.add(self.accum_image, th1)

        # Applica una color map
        # La migliore è COLORMAP_HOT
        # Anche COLORMAP_PINK funziona bene, COLORMAP_BONE è accettabile se il background è nero
        color_image = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_HOT)

        # Sovrappone l'immagine con mappatura dei colori al primo fotogramma
        result_overlay = cv2.addWeighted(self.first_frame, 0.7, color_image, 0.7, 0)

        self.result_overlay = result_overlay

