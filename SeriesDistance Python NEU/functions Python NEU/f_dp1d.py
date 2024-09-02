import numpy as np
from scipy.interpolate import interp1d

def f_dp1d(xy, tol=None, numpoints=None, nselimit=None):
    """
    Linienvereinfachung mit einem modifizierten Douglas-Peucker-Algorithmus
    Uwe Ehret, 15. Nov. 2013

    EINGABE
        xy:           (n,2) Matrix mit n x- und y-Daten einer Linie (x in aufsteigender Reihenfolge)
        tol:          maximale Toleranz (in vertikaler, d.h. y-Richtung) zwischen der Original- und der vereinfachten Serie. In Einheiten der y-Werte
        numpoints:    optional: gibt an, mit wie vielen Punkten die Linie approximiert werden soll. Wenn numpoints gesetzt ist, wird tol nicht verwendet
        nselimit:     optional: gibt an, welcher Nash-Sutcliffe-Effizienz zwischen der Original- und der vereinfachten Linie erreicht werden soll.
                      Wenn nselimit gesetzt ist, werden tol und numpoints nicht verwendet

    AUSGABE
        xy_dp:       (m,2) Vektor mit m x- und y-Daten der vereinfachten Linie (x in aufsteigender Reihenfolge) m <= n
                      xy_dp[:,0] --> x-Werte der vereinfachten Serie
                      xy_dp[:,1] --> y-Werte der vereinfachten Serie
    METHODE
        Diese Funktion approximiert eine Linie durch weniger ihrer Punkte
        Die vereinfachte Linie beginnt mit dem ersten und dem letzten Punkt von xy, dann werden die Punkte mit dem maximalen Abstand sukzessive hinzugefügt, bis
        das ausgewählte Abbruchkriterium erfüllt ist (tol, numpoints oder nselimit)
    """

    # Überprüfen der Eingabe
    if nselimit is not None:
        calccase = 3
    elif numpoints is not None:
        calccase = 2
    else:
        calccase = 1

    # Initialisieren der vereinfachten Linie mit dem ersten und letzten Wert
    xy_dp = np.array([xy[0], xy[-1]])

    alldone = False
    while not alldone:
        # Abtasten der vereinfachten Linie an allen x-Positionen der Original-Linie
        interp_func = interp1d(xy_dp[:, 0], xy_dp[:, 1], kind='linear', fill_value="extrapolate")
        intp_dp = interp_func(xy[:, 0])

        # Finden des maximalen Abstands zwischen der vereinfachten und der Original-Linie
        d = np.abs(xy[:, 1] - intp_dp)  # Bestimmen des absoluten Abstands an jeder x-Position der Original-Linie (vertikal, d.h. in y-Richtung)
        index = np.argmax(d)  # Finden des Maximums der Abstände

        if calccase == 1:  # max tol Limit
            if d[index] > tol:  # wenn der maximale Abstand die Toleranz überschreitet ...
                x_add = xy[index, 0]  # Finden des x-Werts des maximalen Abstands
                y_add = xy[index, 1]  # Finden des y-Werts des maximalen Abstands
                if x_add > xy_dp[-1, 0]:  # wenn der maximale Wert am Ende der Linie liegt (sollte nicht, da wir die ersten und letzten als Anfangswerte nehmen)
                    xy_dp = np.vstack([xy_dp, [x_add, y_add]])
                else:
                    insertpos = np.searchsorted(xy_dp[:, 0], x_add)  # Finden der Position, an der der neue Punkt eingefügt werden soll
                    xy_dp = np.vstack([xy_dp[:insertpos], [x_add, y_add], xy_dp[insertpos:]])  # Hinzufügen des neuen Punkts zur vereinfachten Linie
            else:  # wenn der maximale Abstand kleiner als die Toleranz ist
                alldone = True  # wir sind fertig

        elif calccase == 2:  # numpoints Limit
            if len(xy_dp) < numpoints:  # wenn die Anzahl der Punkte nicht ausreicht ...
                x_add = xy[index, 0]  # Finden des x-Werts des maximalen Abstands
                y_add = xy[index, 1]  # Finden des y-Werts des maximalen Abstands
                if x_add > xy_dp[-1, 0]:  # wenn der maximale Wert am Ende der Linie liegt (sollte nicht, da wir die ersten und letzten als Anfangswerte nehmen)
                    xy_dp = np.vstack([xy_dp, [x_add, y_add]])
                else:
                    insertpos = np.searchsorted(xy_dp[:, 0], x_add)  # Finden der Position, an der der neue Punkt eingefügt werden soll
                    xy_dp = np.vstack([xy_dp[:insertpos], [x_add, y_add], xy_dp[insertpos:]])  # Hinzufügen des neuen Punkts zur vereinfachten Linie
            else:  # wenn die vereinfachte Linie aus numpoints besteht
                alldone = True  # wir sind fertig

        elif calccase == 3:  # nse Limit
            # Berechnen der NSE
            A = xy[:, 0]
            B = intp_dp
            C = np.column_stack((A, B))
            nse, metric_id = f_nashsutcliffe(xy, C)
            ########################### FUNKTION F_NASHSUTCLIFFE IST NICHT VORHANDEN!!!!! ###########################

            if nse < nselimit:  # die NSE ist noch nicht gut genug ...
                x_add = xy[index, 0]  # Finden des x-Werts des maximalen Abstands
                y_add = xy[index, 1]  # Finden des y-Werts des maximalen Abstands
                if x_add > xy_dp[-1, 0]:  # wenn der maximale Wert am Ende der Linie liegt (sollte nicht, da wir die ersten und letzten als Anfangswerte nehmen)
                    xy_dp = np.vstack([xy_dp, [x_add, y_add]])
                else:
                    insertpos = np.searchsorted(xy_dp[:, 0], x_add)  # Finden der Position, an der der neue Punkt eingefügt werden soll
                    xy_dp = np.vstack([xy_dp[:insertpos], [x_add, y_add], xy_dp[insertpos:]])  # Hinzufügen des neuen Punkts zur vereinfachten Linie
            else:  # wenn die NSE gut genug ist
                alldone = True  # wir sind fertig

    return xy_dp