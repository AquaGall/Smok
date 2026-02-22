# curves/ulam.py

def ulam_coordinates(n):
    """
    Zwraca współrzędne (x, y) liczby n w spirali Ulama.
    Spirala rośnie w prawo, potem w górę, w lewo, w dół, itd.
    """

    if n < 1:
        raise ValueError("n must be >= 1")

    # warstwa spirali (pierścień kwadratowy)
    layer = 0
    while (2*layer + 1)**2 < n:
        layer += 1

    # największa liczba w tej warstwie
    max_val = (2*layer + 1)**2
    side_len = 2 * layer

    # start: prawy dolny róg warstwy
    x = layer
    y = -layer

    # ile kroków od max_val do n
    diff = max_val - n

    # cztery boki spirali
    if diff < side_len:
        # dół: idziemy w lewo
        x -= diff
    elif diff < 2 * side_len:
        # lewo: idziemy w górę
        x -= side_len
        y += diff - side_len
    elif diff < 3 * side_len:
        # góra: idziemy w prawo
        x -= side_len - (diff - 2 * side_len)
        y += side_len
    else:
        # prawo: idziemy w dół
        y += side_len - (diff - 3 * side_len)

    return (x, y)
