import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- NOWE IMPORTY DLA PLOTLY ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.fabrik_engine import fabrik, compute_angles
from core.prime_utils import is_prime
from curves.ulam import ulam_coordinates

from analysis.angles import build_angles_df, plot_angles
from analysis.delta_angles import compute_delta, plot_delta
from analysis.fft_analysis import extract_real_signal, compute_fft_signal, plot_fft_single
from analysis.heatmap import plot_heatmap


st.set_page_config(layout="wide")

st.title("🔵 FABRIK – odcisk palca liczb pierwszych na spirali Ulama")

# --- ZMIANA: dodajemy nową opcję animacji ---
mode = st.sidebar.radio(
    "Tryb pracy:",
    [
        "Obliczenia (tabela + wykresy)",
        "Animacja (matplotlib – wolna)",
        "Animacja (Plotly – szybka)"
    ]
)

n_min = st.sidebar.number_input("n min", value=2)
n_max = st.sidebar.number_input("n max", value=200)

R = 1.0
L = 2.0
max_iters = st.sidebar.slider("Iteracje FABRIK", 10, 200, 50)

if mode == "Animacja (matplotlib – wolna)":
    anim_speed = st.sidebar.slider("Prędkość animacji", 0.01, 1.0, 0.1)

start = st.sidebar.button("▶️ Start")


# ------------------ TRYB OBLICZENIA ------------------

if start and mode.startswith("Obliczenia"):

    df = build_angles_df(
        n_min=n_min,
        n_max=n_max,
        ulam_coordinates=ulam_coordinates,
        fabrik=fabrik,
        compute_angles=compute_angles,
        L=L,
        R=R,
        max_iters=max_iters,
        is_prime=is_prime,
    )

    st.subheader("📄 Tabela kątów")
    st.dataframe(df, use_container_width=True)

    angle_cols = [c for c in df.columns if c.startswith("K")][:3]

    primes_df = df[df["prime"] == True]
    comp_df = df[df["prime"] == False]

    # K
    st.subheader("📈 Wykresy kątów K1, K2, K3")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.pyplot(plot_angles(df, angle_cols, "Wszystkie liczby"))
    with c2:
        if len(primes_df) > 0:
            st.pyplot(plot_angles(primes_df, angle_cols, "Liczby pierwsze"))
        else:
            st.write("Brak liczb pierwszych.")
    with c3:
        if len(comp_df) > 0:
            st.pyplot(plot_angles(comp_df, angle_cols, "Liczby złożone"))
        else:
            st.write("Brak liczb złożonych.")

    # ΔK
    st.subheader("📄 Tabela różnic kątów ΔK")
    delta_df = compute_delta(df, angle_cols)
    st.dataframe(delta_df, use_container_width=True)

    primes_delta = delta_df[delta_df["prime"] == True]
    comp_delta = delta_df[delta_df["prime"] == False]

    st.subheader("📈 Wykresy różnic kątów ΔK1, ΔK2, ΔK3")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.pyplot(plot_delta(delta_df, angle_cols, "ΔK – wszystkie liczby"))
    with d2:
        if len(primes_delta) > 0:
            st.pyplot(plot_delta(primes_delta, angle_cols, "ΔK – pierwsze"))
        else:
            st.write("Brak liczb pierwszych.")
    with d3:
        if len(comp_delta) > 0:
            st.pyplot(plot_delta(comp_delta, angle_cols, "ΔK – złożone"))
        else:
            st.write("Brak liczb złożonych.")

    # HEATMAPY
    st.subheader("🔥 Heatmapa kątów K1–K3")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.pyplot(plot_heatmap(df, angle_cols, "Heatmapa K – wszystkie"))
    with h2:
        if len(primes_df) > 0:
            st.pyplot(plot_heatmap(primes_df, angle_cols, "Heatmapa K – pierwsze"))
        else:
            st.write("Brak liczb pierwszych.")
    with h3:
        if len(comp_df) > 0:
            st.pyplot(plot_heatmap(comp_df, angle_cols, "Heatmapa K – złożone"))
        else:
            st.write("Brak liczb złożonych.")

    st.subheader("🔥 Heatmapa różnic kątów ΔK1–ΔK3")
    hd1, hd2, hd3 = st.columns(3)
    with hd1:
        st.pyplot(plot_heatmap(delta_df, angle_cols, "Heatmapa ΔK – wszystkie"))
    with hd2:
        if len(primes_delta) > 0:
            st.pyplot(plot_heatmap(primes_delta, angle_cols, "Heatmapa ΔK – pierwsze"))
        else:
            st.write("Brak liczb pierwszych.")
    with hd3:
        if len(comp_delta) > 0:
            st.pyplot(plot_heatmap(comp_delta, angle_cols, "Heatmapa ΔK – złożone"))
        else:
            st.write("Brak liczb złożonych.")

    # FFT K
    st.subheader("📈 FFT kątów (K1, K2, K3)")
    fk1, fk2, fk3 = st.columns(3)

    with fk1:
        sig_all = extract_real_signal(df, angle_cols[0])
        sig_primes = extract_real_signal(primes_df, angle_cols[0]) if len(primes_df) > 0 else np.array([])
        sig_comp = extract_real_signal(comp_df, angle_cols[0]) if len(comp_df) > 0 else np.array([])

        freqs, mags = compute_fft_signal(sig_all)
        st.pyplot(plot_fft_single(freqs, mags, "FFT K1 – wszystkie"))
        if len(sig_primes) > 1:
            freqs, mags = compute_fft_signal(sig_primes)
            st.pyplot(plot_fft_single(freqs, mags, "FFT K1 – pierwsze"))
        if len(sig_comp) > 1:
            freqs, mags = compute_fft_signal(sig_comp)
            st.pyplot(plot_fft_single(freqs, mags, "FFT K1 – złożone"))

    with fk2:
        sig_all = extract_real_signal(df, angle_cols[1])
        sig_primes = extract_real_signal(primes_df, angle_cols[1]) if len(primes_df) > 0 else np.array([])
        sig_comp = extract_real_signal(comp_df, angle_cols[1]) if len(comp_df) > 0 else np.array([])

        freqs, mags = compute_fft_signal(sig_all)
        st.pyplot(plot_fft_single(freqs, mags, "FFT K2 – wszystkie"))
        if len(sig_primes) > 1:
            freqs, mags = compute_fft_signal(sig_primes)
            st.pyplot(plot_fft_single(freqs, mags, "FFT K2 – pierwsze"))
        if len(sig_comp) > 1:
            freqs, mags = compute_fft_signal(sig_comp)
            st.pyplot(plot_fft_single(freqs, mags, "FFT K2 – złożone"))

    with fk3:
        sig_all = extract_real_signal(df, angle_cols[2])
        sig_primes = extract_real_signal(primes_df, angle_cols[2]) if len(primes_df) > 0 else np.array([])
        sig_comp = extract_real_signal(comp_df, angle_cols[2]) if len(comp_df) > 0 else np.array([])

        freqs, mags = compute_fft_signal(sig_all)
        st.pyplot(plot_fft_single(freqs, mags, "FFT K3 – wszystkie"))
        if len(sig_primes) > 1:
            freqs, mags = compute_fft_signal(sig_primes)
            st.pyplot(plot_fft_single(freqs, mags, "FFT K3 – pierwsze"))
        if len(sig_comp) > 1:
            freqs, mags = compute_fft_signal(sig_comp)
            st.pyplot(plot_fft_single(freqs, mags, "FFT K3 – złożone"))

    # FFT ΔK
    st.subheader("📈 FFT różnic kątów (ΔK1, ΔK2, ΔK3)")
    fd1, fd2, fd3 = st.columns(3)

    with fd1:
        sig_all = extract_real_signal(delta_df, angle_cols[0])
        sig_primes = extract_real_signal(primes_delta, angle_cols[0]) if len(primes_delta) > 0 else np.array([])
        sig_comp = extract_real_signal(comp_delta, angle_cols[0]) if len(comp_delta) > 0 else np.array([])

        freqs, mags = compute_fft_signal(sig_all)
        st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK1 – wszystkie"))
        if len(sig_primes) > 1:
            freqs, mags = compute_fft_signal(sig_primes)
            st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK1 – pierwsze"))
        if len(sig_comp) > 1:
            freqs, mags = compute_fft_signal(sig_comp)
            st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK1 – złożone"))

    with fd2:
        sig_all = extract_real_signal(delta_df, angle_cols[1])
        sig_primes = extract_real_signal(primes_delta, angle_cols[1]) if len(primes_delta) > 0 else np.array([])
        sig_comp = extract_real_signal(comp_delta, angle_cols[1]) if len(comp_delta) > 0 else np.array([])

        freqs, mags = compute_fft_signal(sig_all)
        st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK2 – wszystkie"))
        if len(sig_primes) > 1:
            freqs, mags = compute_fft_signal(sig_primes)
            st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK2 – pierwsze"))
        if len(sig_comp) > 1:
            freqs, mags = compute_fft_signal(sig_comp)
            st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK2 – złożone"))

    with fd3:
        sig_all = extract_real_signal(delta_df, angle_cols[2])
        sig_primes = extract_real_signal(primes_delta, angle_cols[2]) if len(primes_delta) > 0 else np.array([])
        sig_comp = extract_real_signal(comp_delta, angle_cols[2]) if len(comp_delta) > 0 else np.array([])

        freqs, mags = compute_fft_signal(sig_all)
        st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK3 – wszystkie"))
        if len(sig_primes) > 1:
            freqs, mags = compute_fft_signal(sig_primes)
            st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK3 – pierwsze"))
        if len(sig_comp) > 1:
            freqs, mags = compute_fft_signal(sig_comp)
            st.pyplot(plot_fft_single(freqs, mags, "FFT ΔK3 – złożone"))


# ------------------ TRYB ANIMACJA MATPLOTLIB (TWÓJ ORYGINAŁ) ------------------

if start and mode == "Animacja (matplotlib – wolna)":

    plot_area = st.empty()

    spiral_x = []
    spiral_y = []
    spiral_color = []

    for m in range(int(n_min), int(n_max) + 1):
        x_m, y_m = ulam_coordinates(m)
        spiral_x.append(x_m)
        spiral_y.append(y_m)
        spiral_color.append("green" if is_prime(m) else "gray")

    for n in range(int(n_min), int(n_max) + 1):
        x, y = ulam_coordinates(n)
        target = np.array([x, y])

        dist = np.sqrt(x * x + y * y)
        k = int(np.ceil(dist / L))

        centers = [np.array([0.0, 0.0])]
        for _ in range(k):
            centers.append(centers[-1] + np.array([L, 0]))

        centers = fabrik(centers, target, L, R, max_iters=max_iters)

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(spiral_x, spiral_y, c=spiral_color, s=30, alpha=0.6)
        ax.scatter([x], [y], color="red", s=200)

        for i in range(len(centers) - 1):
            ax.plot(
                [centers[i][0], centers[i+1][0]],
                [centers[i][1], centers[i+1][1]],
                "k-",
                linewidth=2,
            )

        for c in centers:
            circ = plt.Circle((c[0], c[1]), R, fill=False, color="blue")
            ax.add_patch(circ)

        ax.set_title(f"n = {n}   prime = {is_prime(n)}   target = ({x}, {y})")
        ax.set_aspect("equal")
        ax.grid(True)

        plot_area.pyplot(fig)

        time.sleep(anim_speed)


# ------------------ NOWA ANIMACJA PLOTLY – SZYBKA ------------------

if start and mode == "Animacja (Plotly – szybka)":

    st.subheader("⏳ Proszę czekać… generuję animację (może to potrwać kilka sekund)")

    # Przygotowanie spirali Ulama
    spiral_x = []
    spiral_y = []
    spiral_color = []

    for m in range(int(n_min), int(n_max) + 1):
        x_m, y_m = ulam_coordinates(m)
        spiral_x.append(x_m)
        spiral_y.append(y_m)
        spiral_color.append("green" if is_prime(m) else "gray")

    # Najpierw ustalamy MAKSYMALNĄ liczbę segmentów (k) w całym zakresie,
    # żeby każda klatka miała tyle samo trace’ów z okręgami.
    max_k = 0
    for n in range(int(n_min), int(n_max) + 1):
        x, y = ulam_coordinates(n)
        dist = np.sqrt(x * x + y * y)
        k = int(np.ceil(dist / L))
        if k > max_k:
            max_k = k

    max_centers = max_k + 1  # bo centers ma k+1 punktów (łącznie z początkiem)

    frames = []
    theta = np.linspace(0, 2 * np.pi, 60)

    for n in range(int(n_min), int(n_max) + 1):
        x, y = ulam_coordinates(n)
        target = np.array([x, y])

        dist = np.sqrt(x * x + y * y)
        k = int(np.ceil(dist / L))

        centers = [np.array([0.0, 0.0])]
        for _ in range(k):
            centers.append(centers[-1] + np.array([L, 0]))

        centers = fabrik(centers, target, L, R, max_iters=max_iters)

        seg_x = [c[0] for c in centers]
        seg_y = [c[1] for c in centers]

        # Koła wokół segmentów – STAŁA liczba trace’ów = max_centers
        circle_traces = []
        for idx in range(max_centers):
            if idx < len(centers):
                cx = centers[idx][0]
                cy = centers[idx][1]
                circle_traces.append(
                    go.Scatter(
                        x=cx + R * np.cos(theta),
                        y=cy + R * np.sin(theta),
                        mode="lines",
                        line=dict(color="blue", width=1),
                        showlegend=False
                    )
                )
            else:
                # „Pusty” okrąg – niewidoczny, ale trace istnieje
                circle_traces.append(
                    go.Scatter(
                        x=[np.nan],
                        y=[np.nan],
                        mode="lines",
                        line=dict(color="blue", width=1),
                        showlegend=False
                    )
                )

        frame_data = [
            go.Scatter(
                x=spiral_x,
                y=spiral_y,
                mode="markers",
                marker=dict(color=spiral_color, size=6),
                name="Spirala"
            ),
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(color="red", size=12),
                name="Cel"
            ),
            go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="lines+markers",
                line=dict(color="black", width=3),
                marker=dict(size=8),
                name="Smok"
            )
        ] + circle_traces

        frames.append(go.Frame(data=frame_data, name=str(n)))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    fig.update_layout(
        width=800,
        height=800,
        title="Animacja FABRIK na spirali Ulama (Plotly – szybka)",
        xaxis=dict(scaleanchor="y", showgrid=True),
        yaxis=dict(showgrid=True),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="▶ Start",
                        method="animate",
                        args=[None, {"frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True}]
                    ),
                    dict(
                        label="⏸ Stop",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    )
                ]
            )
        ],

        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[str(n)], {"frame": {"duration": 0, "redraw": True}}],
                        label=str(n)
                    )
                    for n in range(int(n_min), int(n_max) + 1)
                ],
                currentvalue={"prefix": "n = "}
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

