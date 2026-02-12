import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.patches import Arc, Rectangle, Circle

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Scouting Wyscout Pro")
st.title("⚽ Scouting & Role Scoring Engine")

# ==========================================================
# CONFIG ROLES (TEMPORAL DEMO)
# ==========================================================
# ⚠️ REEMPLAZA ESTO CON TU DICCIONARIO REAL
pesos_roles_mejorados = {

 # ==============================
    #          PORTERO
    # ==============================
    "Portero": {
        "Goles recibidos/90": 0.20,
        "xG en contra/90": 0.18,
        "Goles evitados/90": 0.20,
        "Paradas, %": 0.18,
        "Remates en contra/90": 0.10,
        "Porterías imbatidas en los 90": 0.08,
        "Duelos aéreos en los 90": 0.03,
        "Pases hacia adelante/90": 0.015,
        "Precisión pases, %": 0.01,
        "Pases hacia atrás recibidos del arquero/90": 0.005
    },

    "Portero_Avanzado": {
        "Goles evitados/90": 0.22,
        "Paradas, %": 0.16,
        "xG en contra/90": 0.14,
        "Salidas/90": 0.12,
        "Duelos aéreos en los 90": 0.10,
        "Pases largos/90": 0.09,
        "Precisión pases largos, %": 0.07,
        "Longitud media pases largos, m": 0.04,
        "Pases progresivos/90": 0.04,
        "Pases hacia adelante/90": 0.02
    },

    # ==============================
    #      LATERAL DEFENSIVO
    # ==============================
    "Lateral_Defensivo": {
        "Acciones defensivas realizadas/90": 0.20,
        "Duelos defensivos/90": 0.12,
        "Duelos defensivos ganados, %": 0.12,
        "Entradas/90": 0.12,
        "Posesión conquistada después de una entrada": 0.08,
        "Interceptaciones/90": 0.10,
        "Tiros interceptados/90": 0.08,
        "Faltas/90": 0.06,
        "Carreras en progresión/90": 0.06,
        "Pases/90": 0.03,
        "Precisión pases, %": 0.02,
        "Pases hacia adelante/90": 0.01
    },

    # ==============================
    #      LATERAL OFENSIVO
    # ==============================
    "Lateral_Ofensivo": {
        "Carreras en progresión/90": 0.18,
        "Aceleraciones/90": 0.10,
        "Regates/90": 0.14,
        "Regates realizados, %": 0.08,
        "Centros/90": 0.14,
        "Precisión centros, %": 0.08,
        "Pases progresivos/90": 0.10,
        "Pases al área de penalti/90": 0.08,
        "Pases en el último tercio/90": 0.06,
        "Toques en el área de penalti/90": 0.04
    },

    # ==============================
    #      CENTRAL STOPPER
    # ==============================
    "Central_Stopper": {
        "Duelos defensivos/90": 0.18,
        "Duelos defensivos ganados, %": 0.16,
        "Duelos aéreos en los 90": 0.16,
        "Duelos aéreos ganados, %": 0.12,
        "Entradas/90": 0.10,
        "Interceptaciones/90": 0.12,
        "Tiros interceptados/90": 0.10,
        "Posesión conquistada después de una entrada": 0.04,
        "Faltas/90": 0.02
    },

    # ==============================
    #      CENTRAL CLÁSICO
    # ==============================
    "Central_Clasico": {
        "Duelos defensivos ganados, %": 0.16,
        "Duelos aéreos ganados, %": 0.14,
        "Interceptaciones/90": 0.14,
        "Tiros interceptados/90": 0.12,
        "Pases hacia adelante/90": 0.10,
        "Pases largos/90": 0.10,
        "Precisión pases largos, %": 0.08,
        "Longitud media pases, m": 0.08,
        "Pases progresivos/90": 0.08
    },

    # ==============================
    #        CENTRAL SALIDA
    # ==============================
    "Central_Salida": {
        "Pases/90": 0.12,
        "Precisión pases, %": 0.10,
        "Pases hacia adelante/90": 0.12,
        "Pases progresivos/90": 0.18,
        "Precisión pases progresivos, %": 0.14,
        "Pases largos/90": 0.10,
        "Precisión pases largos, %": 0.10,
        "Longitud media pases largos, m": 0.08,
        "Carreras en progresión/90": 0.06
    },

    # ==============================
    #     PIVOTE DEFENSIVO
    # ==============================
    "Pivote_Defensivo": {
        "Acciones defensivas realizadas/90": 0.20,
        "Duelos defensivos/90": 0.12,
        "Duelos defensivos ganados, %": 0.12,
        "Entradas/90": 0.12,
        "Interceptaciones/90": 0.12,
        "Tiros interceptados/90": 0.10,
        "Pases cortos / medios /90": 0.10,
        "Precisión pases cortos / medios, %": 0.06,
        "Pases progresivos/90": 0.04,
        "Pases hacia adelante/90": 0.02
    },

    # ==============================
    #          INTERIOR
    # ==============================
    "Interior": {
        "Pases en el último tercio/90": 0.16,
        "Precisión pases en el último tercio, %": 0.12,
        "Pases progresivos/90": 0.18,
        "Pases al área de penalti/90": 0.10,
        "xA/90": 0.12,
        "Jugadas claves/90": 0.10,  # antes Key passes/90
        "Regates/90": 0.08,
        "Carreras en progresión/90": 0.08,
        "Acciones de ataque exitosas/90": 0.06
    },

    # ==============================
    #         BOX TO BOX
    # ==============================
    "Box_to_Box": {
        "Carreras en progresión/90": 0.18,
        "Aceleraciones/90": 0.12,
        "Remates/90": 0.12,  # antes Tiros/90
        "Goles/90": 0.12,
        "xG/90": 0.10,
        "Pases progresivos/90": 0.12,
        "Duelos/90": 0.08,
        "Duelos ganados, %": 0.06,
        "Interceptaciones/90": 0.06,
        "Acciones de ataque exitosas/90": 0.04
    },

    # ==============================
    #         MEDIAPUNTA
    # ==============================
    "Mediapunta": {
        "Asistencias/90": 0.16,
        "xA/90": 0.14,
        "Jugadas claves/90": 0.12,  # antes Key passes/90
        "Regates/90": 0.10,
        "Regates realizados, %": 0.06,
        "Remates/90": 0.10,  # antes Tiros/90
        "Goles/90": 0.10,
        "xG/90": 0.10,
        "Jugadas claves/90": 0.06,
        "Pases en el último tercio/90": 0.04,
        "Toques en el área de penalti/90": 0.02
    },

    # ==============================
    #    EXTREMO ASOCIATIVO
    # ==============================
    "Extremo_Asociativo": {
        "Asistencias/90": 0.20,
        "xA/90": 0.16,
        "Jugadas claves/90": 0.14,  # antes Key passes/90
        "Pases progresivos/90": 0.12,
        "Centros/90": 0.10,
        "Precisión centros, %": 0.08,
        "Pases en el último tercio/90": 0.08,
        "Regates/90": 0.06,
        "Regates realizados, %": 0.04,
        "Pases al área de penalti/90": 0.02
    },

    # ==============================
    #        EXTREMO PURO
    # ==============================
    "Extremo_Puro": {
        "Regates/90": 0.20,
        "Regates realizados, %": 0.12,
        "Duelos atacantes/90": 0.12,
        "Duelos atacantes ganados, %": 0.10,
        "Carreras en progresión/90": 0.14,
        "Aceleraciones/90": 0.10,
        "Centros/90": 0.10,
        "Precisión centros, %": 0.08,
        "Remates/90": 0.02,  # antes Tiros/90
        "Goles/90": 0.01,
        "xG/90": 0.01
    },

    # ==============================
    #     DELANTERO GOLEADOR
    # ==============================
    "Delantero_Goleador": {
        "Goles": 0.20,
        "Goles/90": 0.18,
        "xG/90": 0.16,
        "Remates/90": 0.14,  # antes Tiros/90
        "Tiros a la portería, %": 0.10,
        "Goles hechos, %": 0.10,
        "Toques en el área de penalti/90": 0.06,
        "Carreras en progresión/90": 0.04,
        "Pases al área de penalti/90": 0.02
    },

    # ==============================
    #       DELANTERO MÓVIL
    # ==============================
    "Delantero_Movil": {
        "Goles/90": 0.16,
        "xG/90": 0.14,
        "Asistencias/90": 0.14,
        "xA/90": 0.12,
        "Regates/90": 0.10,
        "Carreras en progresión/90": 0.10,
        "Duelos atacantes/90": 0.06,
        "Pases progresivos/90": 0.06,
        "Pases al área de penalti/90": 0.06,
        "Toques en el área de penalti/90": 0.06
    }

}

metricas_negativas = [

    'Goles recibidos/90',
    'xG en contra/90',
    'Remates en contra/90',
    'Faltas/90',
]

roles_metrics = {
    rol: list(metrics.keys())
    for rol, metrics in pesos_roles_mejorados.items()
}
# ==========================================================
# POSICIONES — NORMALIZACIÓN
# ==========================================================

pos_equivalencias = {
    "GK": ["GK"],

    "CB": ["CB","RCB","LCB"],
    "RB": ["RB","RWB"],
    "LB": ["LB","LWB"],

    "DM": ["DMF","LDMF","RDMF"],
    "CM": ["CMF","LCMF","RCMF","MF"],
    "AM": ["AMF","LAMF","RAMF","CAM","AM"],

    "RW": ["RW","RWF"],
    "LW": ["LW","LWF"],

    "FW": ["FW","CF","ST","S"]
}

rol_pos_map = {
    "Portero": ["GK"],
    "Portero_Avanzado": ["GK"],

    "Lateral_Defensivo": ["RB","LB"],
    "Lateral_Ofensivo": ["RB","LB"],

    "Central_Clasico": ["CB"],
    "Central_Salida": ["CB"],
    "Central_Stopper": ["CB"],

    "Pivote_Defensivo": ["DM"],
    "Interior": ["CM","AM"],
    "Box_to_Box": ["CM"],
    "Mediapunta": ["AM"],

    "Extremo_Asociativo": ["RW","LW"],
    "Extremo_Puro": ["RW","LW"],

    "Delantero_Movil": ["FW"],
    "Delantero_Goleador": ["FW"]
}

def normalize_positions(pos_string):
    if pd.isna(pos_string):
        return []

    tokens = [p.strip().upper() for p in str(pos_string).split(",")]
    categorias = set()

    for token in tokens:
        for categoria, equivalencias in pos_equivalencias.items():
            if token in equivalencias:
                categorias.add(categoria)

    return list(categorias)
# ==========================================================
# FUNCIONES
# ==========================================================

@st.cache_data
@st.cache_data
def load_data(files):
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if "Posición específica" in df.columns:
        df["Pos_norm"] = df["Posición específica"].apply(normalize_positions)
    else:
        df["Pos_norm"] = [[] for _ in range(len(df))]

    return df


def percentile_normalization(data, metrics):
    df = data.copy()

    for metric in metrics:
        if metric not in df.columns:
            continue

        values = df[metric].dropna()
        if len(values) < 2:
            df[metric] = 0.5
            continue

        p5, p95 = np.percentile(values, [5, 95])
        if p95 - p5 == 0:
            df[metric] = 0.5
            continue

        if metric in metricas_negativas:
            df[metric] = (p95 - df[metric]) / (p95 - p5)
        else:
            df[metric] = (df[metric] - p5) / (p95 - p5)

        df[metric] = df[metric].clip(0, 1)

    return df


def compute_role_scores(players, min_minutes):
    role_scores = {}

    for rol, weights in pesos_roles_mejorados.items():

        df = players[players["Minutos jugados"] >= min_minutes].copy()

        # 🔹 FILTRO POSICIÓN
        allowed_positions = rol_pos_map.get(rol, [])
        if "Pos_norm" in df.columns:
            df = df[
                df["Pos_norm"].apply(
                    lambda lst: any(p in allowed_positions for p in lst)
                )
            ]

        if df.empty:
            continue

        metrics = [m for m in weights if m in df.columns]
        if not metrics:
            continue

        df_norm = percentile_normalization(df, metrics)

        scores = []
        for _, row in df_norm.iterrows():
            score = sum(row[m] * weights[m] for m in metrics if not pd.isna(row[m]))
            scores.append(score)

        df["Rating"] = np.round(np.array(scores) * 10, 2)
        role_scores[rol] = df.sort_values("Rating", ascending=False)

    return role_scores



def radar_plot(df, role, players_selected):
    metrics = roles_metrics[role]
    df_norm = percentile_normalization(df, metrics)

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    for player in players_selected:
        row = df_norm[df_norm["Jugador"] == player]
        if row.empty:
            continue
        values = row[metrics].iloc[0].tolist()
        values += values[:1]
        ax.plot(angles, values, label=player)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0,1)
    plt.legend()
    st.pyplot(fig)

def best_player_for_role(role_scores, role, used_players):
    if role in role_scores and not role_scores[role].empty:

        for _, row in role_scores[role].iterrows():
            jugador = row["Jugador"]

            if jugador not in used_players:
                return jugador

    return "—"


def draw_pitch():
    fig, ax = plt.subplots(figsize=(7, 11))

    # Fondo verde césped
    fig.patch.set_facecolor("#2E7D32")
    ax.set_facecolor("#2E7D32")

    line_color = "white"
    lw = 2

    # Bordes campo
    ax.plot([0, 0], [0, 100], color=line_color, lw=lw)
    ax.plot([0, 100], [100, 100], color=line_color, lw=lw)
    ax.plot([100, 100], [100, 0], color=line_color, lw=lw)
    ax.plot([100, 0], [0, 0], color=line_color, lw=lw)

    # Medio campo
    ax.plot([0, 100], [50, 50], color=line_color, lw=lw)

    # Círculo central
    ax.add_patch(Circle((50, 50), 9, fill=False, color=line_color, lw=lw))
    ax.plot(50, 50, 'o', color=line_color)

    # ÁREAS GRANDES
    ax.add_patch(Rectangle((30, 82), 40, 18, fill=False, ec=line_color, lw=lw))
    ax.add_patch(Rectangle((30, 0), 40, 18, fill=False, ec=line_color, lw=lw))

    # ÁREAS PEQUEÑAS
    ax.add_patch(Rectangle((40, 94), 20, 6, fill=False, ec=line_color, lw=lw))
    ax.add_patch(Rectangle((40, 0), 20, 6, fill=False, ec=line_color, lw=lw))

    # PUNTOS PENALTI
    ax.plot(50, 88, 'o', color=line_color)
    ax.plot(50, 12, 'o', color=line_color)

    # SEMICÍRCULOS ÁREA (LA D BIEN PROPORCIONADA)
    ax.add_patch(
        Arc((50, 84), 14, 14, theta1=200, theta2=340,
            color=line_color, lw=lw)
    )
    ax.add_patch(
        Arc((50, 16), 14, 14, theta1=20, theta2=160,
            color=line_color, lw=lw)
    )

    # CÓRNERS
    r = 3
    ax.add_patch(Arc((0, 0), r*2, r*2, theta1=0, theta2=90,
                     color=line_color, lw=lw))
    ax.add_patch(Arc((100, 0), r*2, r*2, theta1=90, theta2=180,
                     color=line_color, lw=lw))
    ax.add_patch(Arc((0, 100), r*2, r*2, theta1=270, theta2=360,
                     color=line_color, lw=lw))
    ax.add_patch(Arc((100, 100), r*2, r*2, theta1=180, theta2=270,
                     color=line_color, lw=lw))

    # PORTERÍAS
    ax.plot([45, 55], [100, 100], color=line_color, lw=lw)
    ax.plot([45, 55], [0, 0], color=line_color, lw=lw)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    return fig, ax


formation_coords = {
    "4-3-3": {
        "Portero": (50,10),
        "Lateral_Defensivo": (15,30),
        "Lateral_Ofensivo": (85,30),
        "Central_Clasico": (35,30),
        "Central_Salida": (65,30),
        "Pivote_Defensivo": (50,45),
        "Interior": (70,55),
        "Box_to_Box": (30,55),
        "Extremo_Puro": (15,75),
        "Extremo_Asociativo": (85,75),
        "Delantero_Goleador": (50,85)
    },

    "4-4-2": {
        "Portero": (50,10),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Central_Clasico": [(35,30),(65,30)],
        "Pivote_Defensivo": (40,50),
        "Interior": (60,50),
        "Extremo_Puro": [(15,75),(85,75)],
        "Delantero_Goleador": (40,85),
        "Delantero_Movil": (60,85)
    },

    "3-5-2": {
        "Portero": (50,10),
        "Central_Salida": (50,30),
        "Central_Clasico": [(30,30),(70,30)],
        "Lateral_Ofensivo": [(15,50),(85,50)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,85)
    },

    "5-3-2": {
        "Portero": (50,10),
        "Central_Clasico": [(30,30),(70,30)],
        "Central_Salida": (50,30),
        "Lateral_Defensivo": [(15,35),(85,35)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Delantero_Goleador": (40,85),
        "Delantero_Movil": (60,85)
    },

    "4-5-1": {
        "Portero": (50,10),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Central_Clasico": [(35,30),(65,30)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,60),(70,60)],
        "Extremo_Puro": [(15,75),(85,75)],
        "Delantero_Goleador": (50,90)
    },

    "3-4-3": {
        "Portero": (50,10),
        "Central_Salida": (50,30),
        "Central_Clasico": [(30,30),(70,30)],
        "Lateral_Ofensivo": [(15,50),(85,50)],
        "Pivote_Defensivo": (30,55),
        "Interior": [(70,60)],
        "Extremo_Puro": [(15,80),(85,80)],
        "Delantero_Goleador": (50,90)
    }
}
def plot_formation(formacion, alineacion):

    fig, ax = draw_pitch()
    coords_map = formation_coords.get(formacion, {})

    # contador para roles repetidos
    role_counter = {}

    for rol_display, jugador in alineacion:

        rol_base = rol_display.split(" ")[0]
        role_counter[rol_base] = role_counter.get(rol_base, 0)

        coord = coords_map.get(rol_base)

        if isinstance(coord, list):
            # varias coordenadas
            if role_counter[rol_base] < len(coord):
                x, y = coord[role_counter[rol_base]]
            else:
                continue
        else:
            # una sola coordenada
            x, y = coord

        role_counter[rol_base] += 1

        ax.text(
            x, y, jugador,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8)
        )

    st.pyplot(fig)
# ==========================================================
# UI
# ==========================================================

st.sidebar.header("📂 Subir Excel")
files = st.sidebar.file_uploader(
    "Sube archivos",
    type=["xlsx"],
    accept_multiple_files=True
)

if files:

    players = load_data(files)

    if "Minutos jugados" not in players.columns:
        st.error("Falta columna 'Minutos jugados'")
        st.stop()

    min_minutes = st.sidebar.slider(
        "Minutos mínimos",
        0,
        int(players["Minutos jugados"].max()),
        300
    )

    role_scores = compute_role_scores(players, min_minutes)
    
    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs([
        "🏆 Rankings",
        "🕷 Radar",
        "📋 Alineación"
    ])

    # ------------------------------------------------------
    # TAB 1 — RANKINGS
    # ------------------------------------------------------
    with tab1:

        st.subheader("Ranking por Rol")

        if role_scores:
            selected_role = st.selectbox("Rol", list(role_scores.keys()), key="rank")
            df_role = role_scores[selected_role]

            st.dataframe(
                df_role[["Jugador","Equipo","Minutos jugados","Rating"]],
                use_container_width=True
            )
        else:
            st.warning("No hay datos suficientes.")

    # ------------------------------------------------------
    # TAB 2 — RADAR
    # ------------------------------------------------------
    with tab2:

        st.subheader("Radar Comparativo")

        if role_scores:
            selected_role = st.selectbox("Rol Radar", list(role_scores.keys()), key="radar_role")
            df_role = role_scores[selected_role]

            players_list = df_role["Jugador"].tolist()

            selected_players = st.multiselect("Jugadores", players_list)

            if selected_players:
                radar_plot(df_role, selected_role, selected_players)

    # ------------------------------------------------------
    # TAB 3 — ALINEACIÓN
    # ------------------------------------------------------
    with tab3:

        st.subheader("Alineación Automática")

        formacion = st.selectbox(
            "Formación",
            ["4-3-3", "4-4-2","3-5-2","5-3-2","4-5-1","3-4-3"]
        )

        formaciones = {
            "4-3-3": {
                "Portero": 1,
                "Lateral_Defensivo": 1,
                "Lateral_Ofensivo": 1,
                "Central_Clasico": 1,
                "Central_Salida": 1,
                "Pivote_Defensivo": 1,
                "Interior": 1,
                "Box_to_Box": 1,
                "Extremo_Puro": 1,
                "Extremo_Asociativo": 1,
                "Delantero_Goleador": 1
            },
            "4-4-2": {
                "Portero": 1,
                "Lateral_Defensivo": 2,
                "Central_Clasico": 2,
                "Pivote_Defensivo": 1,
                "Interior": 1,
                "Extremo_Puro": 2,
                "Delantero_Goleador": 1,
                "Delantero_Movil": 1
            },
            "3-5-2": {
                "Portero": 1,
                "Central_Salida": 1,
                "Central_Clasico": 2,
                "Lateral_Ofensivo": 2,
                "Pivote_Defensivo": 1,
                "Interior": 2,
                "Delantero_Goleador": 1,
                "Delantero_Movil": 1
            },
            "5-3-2": {
                "Portero": 1,
                "Central_Clasico": 2,
                "Central_Salida": 1,
                "Lateral_Defensivo": 2,
                "Pivote_Defensivo": 1,
                "Interior": 2,
                "Delantero_Goleador": 1,
                "Delantero_Movil": 1
            },
            "4-5-1": {
                "Portero": 1,
                "Lateral_Defensivo": 2,
                "Central_Clasico": 2,
                "Pivote_Defensivo": 1,
                "Interior": 2,
                "Extremo_Puro": 2,
                "Delantero_Goleador": 1
            },
            "3-4-3": {
                "Portero": 1,
                "Central_Salida": 1,
                "Central_Clasico": 2,
                "Lateral_Ofensivo": 2,
                "Pivote_Defensivo": 1,
                "Interior": 1,
                "Extremo_Puro": 2,
                "Delantero_Goleador": 1
            }
        }

        # 🔹 Seguridad: si no existe
        if formacion not in formaciones:
            st.warning("Formación no disponible")
            st.stop()

        used_players = []
        alineacion = []

        for rol, cantidad in formaciones[formacion].items():

            for i in range(cantidad):

                rol_display = f"{rol} {i+1}"

                jugador = best_player_for_role(role_scores, rol, used_players)

                if jugador != "—":
                    used_players.append(jugador)

                alineacion.append((rol_display, jugador))

        st.divider()

        for rol, jugador in alineacion:
            st.write(f"**{rol}** → {jugador}")

        st.divider()

        # 🔹 Solo dibuja si hay coords
        if formacion in formation_coords:
            plot_formation(formacion, alineacion)
        else:
            st.info("No hay coordenadas definidas para esta formación.")
