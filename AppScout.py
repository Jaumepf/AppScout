import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
from matplotlib.patches import Arc, Rectangle, Circle
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Scouting Wyscout Pro")
st.title("⚽ Scouting & Role Scoring Engine v1.1")

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

NORMALIZATION_MODE = "role"  # "role" o "global"

# ==========================================================
# PESOS ROLES (MANTENGO LOS TUYOS SIN MODIFICAR)
# ==========================================================

pesos_roles_mejorados = {

    # ==============================
    # PORTERO
    # ==============================
    "Portero": {
        "Shot_Stop_Index": 0.16,
        "Goles evitados/90": 0.14,
        "Paradas, %": 0.12,
        "xG en contra/90": 0.10,
        "Goles recibidos/90": 0.10,
        "Salidas/90": 0.08,
        "Pases largos/90": 0.08,
        "Precisión pases largos, %": 0.08,
        "Precisión pases, %": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # PORTERO AVANZADO
    # ==============================
    "Portero_Avanzado": {
        "Shot_Stop_Index": 0.14,
        "Goles evitados/90": 0.12,
        "Paradas, %": 0.10,
        "Pases largos/90": 0.12,
        "Precisión pases largos, %": 0.10,
        "Pases progresivos/90": 0.10,
        "Precisión pases, %": 0.10,
        "Salidas/90": 0.08,
        "Verticalidad": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # LATERAL DEFENSIVO
    # ==============================
    "Lateral_Defensivo": {
        "Acciones defensivas realizadas/90": 0.14,
        "Duelos defensivos/90": 0.12,
        "Duelos defensivos ganados, %": 0.12,
        "Interceptaciones/90": 0.10,
        "Entradas/90": 0.10,
        "Recuperaciones": 0.10,
        "Duelo_Eficiencia": 0.08,
        "Faltas/90": 0.08,
        "Discipline_Index": 0.08,
        "Pases/90": 0.08
    },

    # ==============================
    # LATERAL OFENSIVO
    # ==============================
    "Lateral_Ofensivo": {
        "Carreras en progresión/90": 0.14,
        "Aceleraciones/90": 0.12,
        "Regates/90": 0.12,
        "Centros/90": 0.10,
        "Precisión centros, %": 0.10,
        "Pases progresivos/90": 0.10,
        "Verticalidad": 0.08,
        "Threat_Index": 0.08,
        "Recuperaciones": 0.08,
        "Discipline_Index": 0.08
    },

    # ==============================
    # CENTRAL STOPPER
    # ==============================
    "Central_Stopper": {
        "Duelos defensivos/90": 0.14,
        "Duelos defensivos ganados, %": 0.14,
        "Duelos aéreos en los 90": 0.12,
        "Duelos aéreos ganados, %": 0.12,
        "Interceptaciones/90": 0.10,
        "Tiros interceptados/90": 0.10,
        "Entradas/90": 0.08,
        "Recuperaciones": 0.08,
        "Faltas/90": 0.06,
        "Discipline_Index": 0.06
    },

    # ==============================
    # CENTRAL CLÁSICO
    # ==============================
    "Central_Clasico": {
        "Duelos defensivos ganados, %": 0.14,
        "Duelos aéreos ganados, %": 0.12,
        "Interceptaciones/90": 0.12,
        "Tiros interceptados/90": 0.10,
        "Pases hacia adelante/90": 0.10,
        "Pases largos/90": 0.10,
        "Precisión pases largos, %": 0.10,
        "Duelo_Eficiencia": 0.08,
        "Recuperaciones": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # CENTRAL SALIDA
    # ==============================
    "Central_Salida": {
        "Pases/90": 0.14,
        "Precisión pases, %": 0.12,
        "Pases progresivos/90": 0.14,
        "Precisión pases progresivos, %": 0.10,
        "Pases largos/90": 0.10,
        "Precisión pases largos, %": 0.10,
        "Ratio_Pases_Adelante": 0.08,
        "Verticalidad": 0.08,
        "Duelo_Eficiencia": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # PIVOTE DEFENSIVO
    # ==============================
    "Pivote_Defensivo": {
        "Acciones defensivas realizadas/90": 0.14,
        "Interceptaciones/90": 0.12,
        "Entradas/90": 0.12,
        "Recuperaciones": 0.10,
        "Duelos defensivos ganados, %": 0.10,
        "Pases/90": 0.10,
        "Precisión pases, %": 0.08,
        "Pases progresivos/90": 0.08,
        "Faltas/90": 0.08,
        "Discipline_Index": 0.08
    },

    # ==============================
    # INTERIOR
    # ==============================
    "Interior": {
        "Pases en el último tercio/90": 0.14,
        "Precisión pases en el último tercio, %": 0.12,
        "Pases progresivos/90": 0.12,
        "Jugadas claves/90": 0.10,
        "xA/90": 0.10,
        "Verticalidad": 0.10,
        "Threat_Index": 0.08,
        "Regates/90": 0.08,
        "Area_Involvement": 0.08,
        "Discipline_Index": 0.08
    },

    # ==============================
    # BOX TO BOX
    # ==============================
    "Box_to_Box": {
        "Carreras en progresión/90": 0.14,
        "Aceleraciones/90": 0.12,
        "Duelo_Eficiencia": 0.10,
        "Recuperaciones": 0.10,
        "Pases progresivos/90": 0.10,
        "Threat_Index": 0.10,
        "Area_Involvement": 0.08,
        "xG_Overperformance_90": 0.08,
        "Verticalidad": 0.10,
        "Discipline_Index": 0.08
    },

    # ==============================
    # MEDIAPUNTA
    # ==============================
    "Mediapunta": {
        "Jugadas claves/90": 0.16,
        "xA/90": 0.14,
        "xA_Overperformance": 0.12,
        "Pases al área de penalti/90": 0.10,
        "Pases progresivos/90": 0.10,
        "Regates/90": 0.10,
        "Threat_Index": 0.08,
        "Area_Involvement": 0.08,
        "Ratio_Pases_Adelante": 0.06,
        "Discipline_Index": 0.06
    },

    # ==============================
    # EXTREMO ASOCIATIVO
    # ==============================
    "Extremo_Asociativo": {
        "Asistencias/90": 0.14,
        "xA/90": 0.14,
        "Jugadas claves/90": 0.12,
        "Pases progresivos/90": 0.10,
        "Centros/90": 0.10,
        "Precisión centros, %": 0.10,
        "Threat_Index": 0.08,
        "Verticalidad": 0.08,
        "Area_Involvement": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # EXTREMO PURO
    # ==============================
    "Extremo_Puro": {
        "Regates/90": 0.16,
        "Regates realizados, %": 0.12,
        "Duelos atacantes/90": 0.10,
        "Duelos atacantes ganados, %": 0.10,
        "Carreras en progresión/90": 0.12,
        "Aceleraciones/90": 0.10,
        "Threat_Index": 0.10,
        "Area_Involvement": 0.08,
        "xG_Overperformance_90": 0.06,
        "Discipline_Index": 0.06
    },

    # ==============================
    # DELANTERO GOLEADOR
    # ==============================
    "Delantero_Goleador": {
        "Goles/90": 0.16,
        "xG/90": 0.14,
        "Conversion_Gol_%": 0.12,
        "xG_Overperformance_90": 0.12,
        "Remates/90": 0.10,
        "Tiros a la portería, %": 0.10,
        "Area_Involvement": 0.10,
        "Threat_Index": 0.08,
        "Carreras en progresión/90": 0.04,
        "Discipline_Index": 0.04
    },

    # ==============================
    # DELANTERO MÓVIL
    # ==============================
    "Delantero_Movil": {
        "Goles/90": 0.14,
        "xG/90": 0.12,
        "Asistencias/90": 0.12,
        "xA/90": 0.10,
        "Regates/90": 0.10,
        "Carreras en progresión/90": 0.10,
        "Threat_Index": 0.10,
        "Area_Involvement": 0.08,
        "Verticalidad": 0.08,
        "Discipline_Index": 0.06
    }

}


metricas_negativas = [

    # -----------------------
    # DISCIPLINA
    # -----------------------
    'Faltas/90',
    'Tarjetas amarillas',
    'Tarjetas amarillas/90',
    'Tarjetas rojas',
    'Tarjetas rojas/90',

    # -----------------------
    # PORTEROS / DEFENSA GOL
    # -----------------------
    'Goles recibidos',
    'Goles recibidos/90',
    'Remates en contra',
    'Remates en contra/90',
    'xG en contra',
    'xG en contra/90',
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
# NORMALIZACIÓN (RANK PERCENTILE ESTABLE)
# ==========================================================

def percentile_normalization(data, metrics):

    df = data.copy()

    for metric in metrics:

        if metric not in df.columns:
            continue

        if df[metric].dropna().shape[0] < 2:
            df[metric] = 0.5
            continue

        ranks = df[metric].rank(pct=True)

        if metric in metricas_negativas:
            df[metric] = 1 - ranks
        else:
            df[metric] = ranks

        df[metric] = df[metric].clip(0, 1)

    return df

# ==========================================================
# DERIVED METRICS
# ==========================================================

def safe_div(a, b):
    return np.where((b == 0) | (pd.isna(b)), 0, a / b)

def add_derived_metrics(df):

    df["Conversion_Gol_%"] = safe_div(df["Goles"], df["Remates"])
    df["xG_Overperformance_90"] = df["Goles/90"] - df["xG/90"]
    df["xA_Overperformance"] = df["Asistencias"] - df["xA"]

    df["Ratio_Pases_Adelante"] = safe_div(
        df["Pases hacia adelante/90"],
        df["Pases/90"]
    )

    df["Duelo_Eficiencia"] = (
        safe_div(df["Duelos ganados, %"], 100) *
        df["Duelos/90"]
    )

    df["Recuperaciones"] = (
        df["Entradas/90"] +
        df["Interceptaciones/90"]
    )

    df["Threat_Index"] = (
        df["Goles/90"] +
        df["xA/90"] +
        df["Regates/90"] +
        df["Pases al área de penalti/90"]
    )

    df["Verticalidad"] = (
        df["Carreras en progresión/90"] +
        df["Pases progresivos/90"]
    )

    df["Area_Involvement"] = (
        df["Toques en el área de penalti/90"] +
        df["Remates/90"]
    )

    df["Discipline_Index"] = (
        df["Tarjetas amarillas/90"] +
        2 * df["Tarjetas rojas/90"]
    )

    df["Shot_Stop_Index"] = (
        df["Paradas, %"] -
        df["xG en contra/90"]
    )

    return df

# ==========================================================
# SCORING CON PENALIZACIÓN POR MINUTOS
# ==========================================================

def compute_role_scores(players, min_minutes):

    role_scores = {}

    for rol, weights in pesos_roles_mejorados.items():

        df = players.copy()
        df = df[df["Minutos jugados"] >= min_minutes]

        # 🔹 FILTRO POSICIONAL (OBLIGATORIO)
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
            score = sum(
                row[m] * weights[m]
                for m in metrics
                if not pd.isna(row[m])
            )
            scores.append(score)

        df["Rating"] = np.round(np.array(scores) * 10, 2)

        role_scores[rol] = df.sort_values("Rating", ascending=False)

    return role_scores
    
# ==========================================================
# SIMILARIDAD (COSINE SIMILARITY)
# ==========================================================

def find_similar_players(players_df, role, player_name, min_minutes, top_n=5):

    df = players_df.copy()
    df = df[df["Minutos jugados"] >= min_minutes]

    metrics = [m for m in roles_metrics[role] if m in df.columns]
    if not metrics:
        return pd.DataFrame()

    df_norm = percentile_normalization(df, metrics)

    # 🔹 Asegurar numérico
    df_norm[metrics] = df_norm[metrics].apply(
        pd.to_numeric, errors="coerce"
    )

    # 🔹 Reemplazar NaN e infinitos
    df_norm[metrics] = df_norm[metrics] \
        .replace([np.inf, -np.inf], np.nan) \
        .fillna(0)

    target = df_norm[df_norm["Jugador"] == player_name]
    if target.empty:
        return pd.DataFrame()

    matrix = df_norm[metrics].values
    target_vector = target[metrics].values.reshape(1, -1)

    similarities = cosine_similarity(matrix, target_vector).flatten()

    df_norm["Similarity"] = similarities
    df_norm = df_norm[df_norm["Jugador"] != player_name]

    return df_norm.sort_values(
        "Similarity", ascending=False
    ).head(top_n)[
        ["Jugador", "Equipo durante el período seleccionado",
         "Posición específica", "Similarity"]
    ]
# ==========================================================
# FUNCIONES
# ==========================================================

@st.cache_data
def load_data(files):
    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if "Posición específica" in df.columns:
        df["Pos_primary"] = (
        df["Posición específica"]
        .astype(str)
        .apply(lambda x: x.split(",")[0].strip().upper())
    )
        df["Pos_norm"] = df["Posición específica"].apply(normalize_positions)
    else:
        df["Pos_norm"] = [[] for _ in range(len(df))]

    return df

def best_roles_for_player(player_name, players, top_n=3):

    results = []

    for rol, weights in pesos_roles_mejorados.items():

        metrics = [m for m in weights if m in players.columns]
        if not metrics:
            continue

        df_norm = percentile_normalization(players, metrics)
        row = df_norm[df_norm["Jugador"] == player_name]

        if row.empty:
            continue

        row = row.iloc[0]
        score = sum(row[m] * weights[m] for m in metrics if not pd.isna(row[m]))

        results.append((rol, round(score * 10, 2)))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_n]

def top_players_for_role(role_scores, role, top_n=3):

    if role not in role_scores or role_scores[role].empty:
        return []

    df = role_scores[role].head(top_n)

    result = []
    for _, row in df.iterrows():
        result.append((row["Jugador"], row["Rating"]))

    return result

def best_roles_for_player_smart(player_name, players, min_minutes, top_n=3):

    df = players[players["Minutos jugados"] >= min_minutes].copy()
    results = []

    # --- Detectar posición del jugador ---
    player_row = df[df["Jugador"] == player_name]
    if player_row.empty:
        return []

    player_positions = player_row.iloc[0]["Pos_norm"]

    is_gk = "GK" in player_positions

    # --- Definir roles permitidos ---
    if is_gk:
        allowed_roles = ["Portero", "Portero_Avanzado"]
    else:
        allowed_roles = list(pesos_roles_mejorados.keys())

    # --- Calcular scores ---
    for rol in allowed_roles:

        weights = pesos_roles_mejorados[rol]
        metrics = [m for m in weights if m in df.columns]

        if not metrics:
            continue

        df_norm = percentile_normalization(df, metrics)
        row = df_norm[df_norm["Jugador"] == player_name]

        if row.empty:
            continue

        row = row.iloc[0]
        score = sum(row[m] * weights[m] for m in metrics if not pd.isna(row[m]))

        results.append((rol, round(score * 10, 2)))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    # GK solo 2 roles
    if is_gk:
        return results[:2]

    return results[:top_n]

def radar_plot(df, role, players_selected):

    metrics = roles_metrics[role]
    df_norm = percentile_normalization(df, metrics)

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # FONDO CLARO
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f5f5f5")

    colors = ["#1f77b4", "#ff7f0e", "#2ecc71", "#e74c3c"]

    for i, player in enumerate(players_selected):

        row = df_norm[df_norm["Jugador"] == player]
        if row.empty:
            continue

        values = row[metrics].iloc[0].tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2.5,
            color=colors[i % len(colors)],
            label=player
        )

        ax.fill(
            angles,
            values,
            alpha=0.15,
            color=colors[i % len(colors)]
        )

    # ETIQUETAS REALES (sin recortar)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, color="black")

    ax.set_ylim(0, 1)

    # REJILLA SUAVE
    ax.yaxis.grid(True, color="gray", alpha=0.25)
    ax.xaxis.grid(True, color="gray", alpha=0.25)

    # QUITAR NÚMEROS RADIALES
    ax.set_yticklabels([])

    # LEYENDA ABAJO Y VISIBLE
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    # TÍTULO SIMPLE
    plt.title(
        f"Radar comparativo – {role}",
        fontsize=14,
        pad=20
    )

    plt.tight_layout()
    st.pyplot(fig)
    
def radar_vs_role_top_player(players, player_name, role, min_minutes):

    df = players[players["Minutos jugados"] >= min_minutes].copy()

    metrics = roles_metrics[role]
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        return

    df_norm = percentile_normalization(df, metrics)

    player_row = df_norm[df_norm["Jugador"] == player_name]
    if player_row.empty:
        return

    # MEJOR JUGADOR DEL ROL POR SCORE
    weights = pesos_roles_mejorados[role]

    scores = []
    for _, row in df_norm.iterrows():
        s = sum(row[m] * weights[m] for m in metrics if not pd.isna(row[m]))
        scores.append(s)

    df_norm["Score"] = scores
    top_player = df_norm.sort_values("Score", ascending=False).iloc[0]["Jugador"]

    top_row = df_norm[df_norm["Jugador"] == top_player]

    p_vals = player_row.iloc[0][metrics].tolist()
    t_vals = top_row.iloc[0][metrics].tolist()

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    p_vals += p_vals[:1]
    t_vals += t_vals[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")

    ax.plot(angles, p_vals, linewidth=2, label=player_name)
    ax.fill(angles, p_vals, alpha=0.15)

    ax.plot(angles, t_vals, linewidth=2, linestyle="--", label=f"Top {role}")
    ax.fill(angles, t_vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylim(0,1)
    ax.set_yticklabels([])

    ax.grid(alpha=0.25)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()
    st.pyplot(fig)

    
def radar_vs_role_best(players_df, player_name, role):
    metrics = roles_metrics[role]

    df_role = players_df.copy()

    # normalizamos todo el rol
    df_norm = percentile_normalization(df_role, metrics)

    # jugador seleccionado
    player_row = df_norm[df_norm["Jugador"] == player_name]
    if player_row.empty:
        st.warning("Jugador no encontrado")
        return

    player_values = player_row.iloc[0][metrics].tolist()

    # MEJOR VALOR POR MÉTRICA DEL ROL
    role_best = df_norm[metrics].max().tolist()

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    player_values += player_values[:1]
    role_best += role_best[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    # jugador
    ax.plot(angles, player_values, linewidth=2, label=player_name)
    ax.fill(angles, player_values, alpha=0.25)

    # rol ideal
    ax.plot(angles, role_best, linewidth=2, linestyle="--", label="Ideal Rol")
    ax.fill(angles, role_best, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0,1)

    plt.legend(loc="upper right")
    st.pyplot(fig)



def best_player_for_role(role_scores, role, used_players, side=None):

    if role not in role_scores:
        return "—"

    df_full = role_scores[role]

    # Excluir jugadores ya usados
    df_full = df_full[~df_full["Jugador"].isin(used_players)]

    if df_full.empty:
        return "—"

    # =========================
    # FILTRO POR LADO
    # =========================

    if side:

        # ---- EXTREMOS ----
        if role in ["Extremo_Puro", "Extremo_Asociativo"]:

            if side == "left":
                primary_filter = df_full["Pos_primary"].isin(["LW","LWF"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "LW" in x)

            elif side == "right":
                primary_filter = df_full["Pos_primary"].isin(["RW","RWF"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "RW" in x)

        # ---- LATERALES ----
        elif role in ["Lateral_Defensivo", "Lateral_Ofensivo"]:

            if side == "left":
                primary_filter = df_full["Pos_primary"].isin(["LB","LWB"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "LB" in x)

            elif side == "right":
                primary_filter = df_full["Pos_primary"].isin(["RB","RWB"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "RB" in x)

        else:
            primary_filter = None
            norm_filter = None

        # 🥇 1º intento → primaria correcta
        if primary_filter is not None:
            df_primary = df_full[primary_filter]
            if not df_primary.empty:
                return df_primary.iloc[0]["Jugador"]

        # 🥈 2º intento → posición secundaria válida
        if norm_filter is not None:
            df_secondary = df_full[norm_filter]
            if not df_secondary.empty:
                return df_secondary.iloc[0]["Jugador"]

    # 🥉 3º intento → mejor disponible del rol
    return df_full.iloc[0]["Jugador"]
def player_percentiles(players, player_name, role):

    metrics = roles_metrics.get(role, [])
    if not metrics:
        return pd.DataFrame()

    row = players[players["Jugador"] == player_name]
    if row.empty:
        return pd.DataFrame()

    row = row.iloc[0]

    data = []

    for m in metrics:
        if m not in players.columns:
            continue

        distribution = players[m].dropna()
        if len(distribution) == 0:
            continue

        player_value = row[m]
        percentile = (distribution < player_value).mean() * 100

        data.append({
            "Métrica": m,
            "Percentil": round(percentile, 1)
        })

    return pd.DataFrame(data)

def percentile_color(p):

    if p >= 80:
        return "#2ecc71"   # verde
    elif p >= 60:
        return "#3498db"   # azul
    elif p >= 40:
        return "#f1c40f"   # amarillo
    elif p >= 20:
        return "#e67e22"   # naranja
    else:
        return "#e74c3c"   # rojo
    
def plot_percentiles(df_percent):

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [percentile_color(p) for p in df_percent["Percentil"]]

    ax.barh(
        df_percent["Métrica"],
        df_percent["Percentil"],
        color=colors
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentil")
    ax.set_title("Rendimiento por Métrica")

    ax.axvline(20, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(40, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(60, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(80, color="grey", linestyle="--", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)    


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
        "Lateral_Ofensivo": (85,35),
        "Central_Clasico": (35,20),
        "Central_Salida": (65,20),
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
        "Central_Clasico": [(35,20),(65,20)],
        "Pivote_Defensivo": (40,45),
        "Interior": (60,55),
        "Extremo_Puro": [(15,75),(85,75)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,80)
    },

    "3-5-2": {
        "Portero": (50,10),
        "Central_Salida": (50,30),
        "Central_Clasico": [(30,20),(70,20)],
        "Lateral_Ofensivo": [(15,40),(85,40)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,80)
    },

    "5-3-2": {
        "Portero": (50,10),
        "Central_Clasico": [(30,20),(70,20)],
        "Central_Salida": (50,30),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,80)
    },

    "4-5-1": {
        "Portero": (50,10),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Central_Clasico": [(35,20),(65,20)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Extremo_Puro": [(15,75),(85,75)],
        "Delantero_Goleador": (50,90)
    },

    "3-4-3": {
        "Portero": (50,10),
        "Central_Salida": (50,30),
        "Central_Clasico": [(30,20),(70,20)],
        "Lateral_Ofensivo": [(15,50),(85,50)],
        "Pivote_Defensivo": (45,50),
        "Interior": [(55,60)],
        "Extremo_Puro": [(15,80),(85,80)],
        "Delantero_Goleador": (50,90)
    }
}
def plot_formation(formacion, alineacion, role_scores):

    fig, ax = draw_pitch()
    coords_map = formation_coords.get(formacion, {})

    role_counter = {}

    for rol_display, jugador, side in alineacion:

        rol_base = rol_display.split(" ")[0]
        role_counter[rol_base] = role_counter.get(rol_base, 0)

        coord = coords_map.get(rol_base)
        if coord is None:
            continue

        if isinstance(coord, list):
            if role_counter[rol_base] < len(coord):
                x, y = coord[role_counter[rol_base]]
            else:
                continue
        else:
            x, y = coord

        role_counter[rol_base] += 1

        # 🔹 TOP 3 jugadores por lado
        df_role = role_scores.get(rol_base)

        if df_role is None or df_role.empty:
            continue

        df_filtered = df_role.copy()

        # Filtrar por lado si aplica
        if side:

            if rol_base in ["Extremo_Puro", "Extremo_Asociativo"]:

                if side == "left":
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["LW","LWF"])
                    ]
                else:
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["RW","RWF"])
                    ]

            elif rol_base in ["Lateral_Defensivo", "Lateral_Ofensivo"]:

                if side == "left":
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["LB","LWB"])
                    ]
                else:
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["RB","RWB"])
                    ]

        # Si no hay del lado específico → fallback general
        if df_filtered.empty:
            df_filtered = df_role

        top_players = df_filtered.head(3)

        text_lines = [rol_display.replace("_", " ")]

        for i, (_, row) in enumerate(top_players.iterrows(), start=1):
            text_lines.append(f"{i}. {row['Jugador']} ({row['Rating']})")

        final_text = "\n".join(text_lines)

        ax.text(
            x, y, final_text,
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.1,
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                boxstyle="round,pad=0.3",
                edgecolor="black"
            )
        )
    st.pyplot(fig)

def radar_vs_top_player(players_df, role_scores, player_name, role):

    metrics = [m for m in roles_metrics[role] if m in players_df.columns]
    if not metrics:
        return

    df_norm = percentile_normalization(players_df, metrics)

    # jugador seleccionado
    player_row = df_norm[df_norm["Jugador"] == player_name]
    if player_row.empty:
        return

    player_values = player_row.iloc[0][metrics].tolist()

    # TOP REAL DEL ROL
    top_df = role_scores[role]
    top_player_name = top_df.iloc[0]["Jugador"]

    top_row = df_norm[df_norm["Jugador"] == top_player_name]
    top_values = top_row.iloc[0][metrics].tolist()

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    player_values += player_values[:1]
    top_values += top_values[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    ax.plot(angles, player_values, linewidth=2, label=player_name)
    ax.fill(angles, player_values, alpha=0.25)

    ax.plot(angles, top_values, linewidth=2, linestyle="--", label=f"Top {role}")
    ax.fill(angles, top_values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0,1)

    plt.legend(loc="upper right")
    st.pyplot(fig)
    
def scatter_role_universe(df, x_metric, y_metric, highlight_player=None):

    if x_metric not in df.columns or y_metric not in df.columns:
        st.warning("Métrica no disponible")
        return

   
    fig = go.Figure()

    # 🔵 Todos los jugadores (mismo color)
    fig.add_trace(go.Scatter(
        x=df[x_metric],
        y=df[y_metric],
        mode='markers',
        marker=dict(
            size=10,
            color="#1f77b4",
            opacity=0.6
        ),
        text=df["Jugador"],
        hovertemplate=
            "<b>%{text}</b><br>" +
            f"{x_metric}: %{{x}}<br>" +
            f"{y_metric}: %{{y}}<extra></extra>",
        showlegend=False
    ))

    # 🔴 Jugador destacado
    if highlight_player:
        player_row = df[df["Jugador"] == highlight_player]
        if not player_row.empty:
            fig.add_trace(go.Scatter(
                x=player_row[x_metric],
                y=player_row[y_metric],
                mode='markers',
                marker=dict(
                    size=16,
                    color="#e74c3c",
                    line=dict(width=2, color="black")
                ),
                text=player_row["Jugador"],
                hovertemplate=
                    "<b>%{text}</b><br>" +
                    f"{x_metric}: %{{x}}<br>" +
                    f"{y_metric}: %{{y}}<extra></extra>",
                showlegend=False
            ))

    # Líneas medias
    fig.add_vline(x=df[x_metric].mean(), line_dash="dash", opacity=0.3)
    fig.add_hline(y=df[y_metric].mean(), line_dash="dash", opacity=0.3)

    fig.update_layout(
        title=f"{x_metric} vs {y_metric}",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)

def get_percentile_color(p):

    if p >= 80:
        return "#2ecc71"   # verde
    elif p >= 60:
        return "#3498db"   # azul
    elif p >= 40:
        return "#f1c40f"   # amarillo
    elif p >= 20:
        return "#e67e22"   # naranja
    else:
        return "#e74c3c"   # rojo



def percentile_to_color(p):
    # Gradiente rojo → amarillo → verde
    if p <= 50:
        r = 231
        g = int(76 + (p/50)*180)
        b = 60
    else:
        r = int(231 - ((p-50)/50)*150)
        g = 200
        b = 60
    return f"rgb({r},{g},{b})"



def stripplot_role_metrics(players_df, role, player_name, min_minutes):

    df = players_df[players_df["Minutos jugados"] >= min_minutes].copy()

    allowed_positions = rol_pos_map.get(role, [])
    df = df[df["Pos_norm"].apply(lambda x: any(p in allowed_positions for p in x))]

    if df.empty:
        st.warning("No hay jugadores en este rol.")
        return

    weights = pesos_roles_mejorados[role]

    # 🔹 Ordenar métricas por peso (importancia del rol)
    metrics = sorted(
        [m for m in weights if m in df.columns],
        key=lambda x: weights[x],
        reverse=True
    )

    player_row = df[df["Jugador"] == player_name]
    if player_row.empty:
        st.warning("Jugador no encontrado en este rol")
        return

    player_row = player_row.iloc[0]

    for metric in metrics:

        values = df[metric].dropna()
        if len(values) == 0:
            continue

        # Ordenamos valores
        values_sorted = values.sort_values().reset_index(drop=True)

        # Percentiles universo
        percentiles = values_sorted.rank(pct=True) * 100
        colors = [percentile_to_color(p) for p in percentiles]

        # 🔥 Beeswarm stacking simétrico
        stack_dict = {}
        y_positions = []

        for v in values_sorted:
            key = round(v, 3)  # agrupar valores cercanos

            if key not in stack_dict:
                stack_dict[key] = 0
            else:
                stack_dict[key] += 1

            level = stack_dict[key]

            # alternar arriba y abajo
            if level % 2 == 0:
                y_positions.append(level * 0.15)
            else:
                y_positions.append(-level * 0.15)

        fig = go.Figure()

        # Universo
        fig.add_trace(go.Scatter(
            x=values_sorted,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=11,
                color=colors,
                opacity=0.9
            ),
            hovertemplate=f"{metric}: %{{x}}<extra></extra>",
            showlegend=False
        ))

        # 🔴 Jugador destacado
        player_value = player_row[metric]
        player_percentile = (values < player_value).mean() * 100

        fig.add_trace(go.Scatter(
            x=[player_value],
            y=[0],
            mode='markers',
            marker=dict(
                size=24,
                color="black",
                line=dict(width=2, color="white")
            ),
            hovertemplate=
                f"<b>{player_name}</b><br>"
                f"{metric}: %{{x}}<br>"
                f"Percentil: {player_percentile:.1f}"
                "<extra></extra>",
            showlegend=False
        ))

        fig.update_layout(
            title=dict(
                text=f"{metric}",
                x=0,
                xanchor="left"
            ),
            yaxis=dict(visible=False),
            xaxis=dict(
                showgrid=False,
                zeroline=False
            ),
            template="simple_white",
            height=180,
            margin=dict(l=40, r=20, t=40, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

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
    players = add_derived_metrics(players)

    # =========================
    # FILTROS
    # =========================

    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filtros")
   
    # MINUTOS 
    min_minutes = st.sidebar.slider(
        "Minutos mínimos",
        0,
        int(players["Minutos jugados"].max()),
        1000
    )

    players_filtered_minutes = players[
        players["Minutos jugados"] >= min_minutes
    ].copy()

    # EDAD
    if "Edad" in players.columns:
        edad_min, edad_max = st.sidebar.slider(
            "Edad",
            int(players["Edad"].min()),
            int(players["Edad"].max()),
            (18, 35)
        )
        players = players[
            (players["Edad"] >= edad_min) &
            (players["Edad"] <= edad_max)
        ]
    # VALOR DE MERCADO
    if "Valor de mercado (Transfermarkt)" in players.columns:

        # Limpiar posibles strings tipo "€10m"
        players["Valor de mercado (Transfermarkt)"] = (
            players["Valor de mercado (Transfermarkt)"]
            .replace('[€,mM]', '', regex=True)
        )

        players["Valor de mercado (Transfermarkt)"] = pd.to_numeric(
            players["Valor de mercado (Transfermarkt)"],
            errors="coerce"
        )

        min_val = float(players["Valor de mercado (Transfermarkt)"].min())
        max_val = float(players["Valor de mercado (Transfermarkt)"].max())

        market_min, market_max = st.sidebar.slider(
            "Valor de mercado (€ millones)",
            min_val,
            max_val,
            (min_val, max_val)
        )

        players = players[
            (players["Valor de mercado (Transfermarkt)"] >= market_min) &
            (players["Valor de mercado (Transfermarkt)"] <= market_max)
        ]
    # VENCIMIENTO CONTRATO
    if "Vencimiento contrato" in players.columns:

        players["Vencimiento contrato"] = pd.to_datetime(
            players["Vencimiento contrato"],
            errors="coerce"
        )

        years_to_expiry = (
            players["Vencimiento contrato"] - pd.Timestamp.today()
        ).dt.days / 365

        max_years = st.sidebar.slider(
            "Contrato vence en ≤ años",
            0.0,
            5.0,
            2.0
        )

        players = players[years_to_expiry <= max_years]
    # PIE
    if "Pie" in players.columns:
        pies = sorted(players["Pie"].dropna().unique())
        pie_sel = st.sidebar.multiselect("Pie dominante", pies)
        if pie_sel:
            players = players[players["Pie"].isin(pie_sel)]

    # COMPETICIÓN
    if "Competición" in players.columns:
        comps = sorted(players["Competición"].dropna().unique())
        comp_sel = st.sidebar.multiselect("Competición", comps)
        if comp_sel:
            players = players[players["Competición"].isin(comp_sel)]

    # AÑO
    if "Año" in players.columns:
        years = sorted(players["Año"].dropna().unique())
        year_sel = st.sidebar.multiselect("Año", years)
        if year_sel:
            players = players[players["Año"].isin(year_sel)]


    # =========================
    # SCORING
    # =========================

    role_scores = compute_role_scores(
        players,
        min_minutes
    )
    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏆 Rankings",
        "🕷 Radar",
        "📋 Alineación",
        "📊 Percentiles",
        "🆚 Comparador",
        "🎯 Role Fit",
        "📍 Scatter",
        "🔎 Similaridad"
    ])



    # ------------------------------------------------------
    # TAB 1 — RANKINGS
    # ------------------------------------------------------
    with tab1:

        st.subheader("Ranking por Rol")

        if role_scores:
            selected_role = st.selectbox("Rol", list(role_scores.keys()), key="rank")
            df_role = role_scores[selected_role]

            # 🔥 Mostrar solo jugadores con Rating válido
            df_role = df_role[df_role["Rating"].notna()]

            st.dataframe(
                df_role[[
                    "Jugador",
                    "Equipo durante el período seleccionado",
                    "Minutos jugados",
                    "Rating"
                ]],
                use_container_width=True
            )

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

                # 🔹 DEFINIR SIDE
                side = None

                if cantidad == 2:
                    side = "left" if i == 0 else "right"

                jugador = best_player_for_role(
                    role_scores,
                    rol,
                    used_players,
                    side=side
                )

                if jugador != "—":
                    used_players.append(jugador)

                alineacion.append((rol_display, jugador, side))

        st.divider()

        for rol, jugador, side in alineacion:
            st.write(f"**{rol}** → {jugador}")

        st.divider()

        # 🔹 Solo dibuja si hay coords
        if formacion in formation_coords:
            plot_formation(formacion, alineacion, role_scores)
        else:
            st.info("No hay coordenadas definidas para esta formación.")
    # ------------------------------------------------------
    # TAB 4 — PERCENTILES
    # ------------------------------------------------------
    with tab4:

        st.subheader("Percentiles por Jugador")

        if role_scores:

            all_players = players["Jugador"].unique().tolist()
            selected_player = st.selectbox(
                "Jugador",
                all_players,
                key="percent_player"
            )

            selected_role = st.selectbox(
                "Rol",
                list(role_scores.keys()),
                key="percent_role"
            )

            df_percent = player_percentiles(
                players,
                selected_player,
                selected_role
            )

            if not df_percent.empty:
                st.dataframe(df_percent, use_container_width=True)

                # mini gráfico
                plot_percentiles(df_percent)
                
            else:
                st.warning("Sin datos para este jugador.")
    # ------------------------------------------------------
    # TAB 5 — COMPARADOR
    # ------------------------------------------------------
    with tab5:

        st.subheader("Comparador de Jugadores")

        if role_scores:

            rol_comp = st.selectbox(
                "Rol",
                list(role_scores.keys()),
                key="comp_role"
            )

            df_role = role_scores[rol_comp]

            jugadores = df_role["Jugador"].tolist()

            jugadores_sel = st.multiselect(
                "Jugadores a comparar",
                jugadores,
                max_selections=10
            )

            if len(jugadores_sel) >= 2:

                metrics = [
                    m for m in roles_metrics[rol_comp]
                    if m in df_role.columns
                ]

                # 🔥 VALORES REALES (NO NORMALIZADOS)
                tabla_real = df_role[
                    df_role["Jugador"].isin(jugadores_sel)
                ][["Jugador"] + metrics]

                st.write("### Métricas reales")
                st.dataframe(tabla_real, use_container_width=True)

            else:
                st.info("Selecciona mínimo 2 jugadores.")
   
    # ------------------------------------------------------
    # TAB 6 — ROLE FIT
    # ------------------------------------------------------
    with tab6:

        st.subheader("🎯 Encaje de jugador por rol")

        if role_scores:

            # SOLO jugadores con minutos mínimos
            eligible_players = players[
                players["Minutos jugados"] >= min_minutes
            ]["Jugador"].unique().tolist()

            player_choice = st.selectbox(
                "Seleccionar jugador",
                eligible_players,
                index=None,
                placeholder="Selecciona un jugador"
            )

            if player_choice:

                top_roles = best_roles_for_player_smart(
                    player_choice,
                    players,
                    min_minutes,
                    top_n=3
                )

                st.markdown("### Mejores roles")

                for rol, score in top_roles:
                    st.write(f"**{rol}** — Rating: {score}")

                st.divider()
                st.markdown("### Comparativa vs Mejor Jugador del Rol")

                cols = st.columns(3)

                for i, (rol, score) in enumerate(top_roles):
                    with cols[i]:
                        st.markdown(f"#### {rol} — {score}")
                        radar_vs_role_top_player(
                            players,
                            player_choice,
                            rol,
                            min_minutes
                        )


        else:
            st.warning("Carga datos primero.")
# ------------------------------------------------------
# TAB 7 — STRIP PLOT POR ROL
# ------------------------------------------------------
    with tab7:

        st.subheader("📊 Distribución por Métricas (Strip Plot)")

        if role_scores:

            # jugador elegible por minutos
            eligible_players = players[
                players["Minutos jugados"] >= min_minutes
            ]["Jugador"].unique().tolist()

            selected_player = st.selectbox(
                "Seleccionar jugador",
                eligible_players
            )

            selected_role = st.selectbox(
                "Seleccionar rol",
                list(role_scores.keys())
            )

            if selected_player and selected_role:

                stripplot_role_metrics(
                    players,
                    selected_role,
                    selected_player,
                    min_minutes
                )
# ------------------------------------------------------
# TAB 8 — Similaridad
# ------------------------------------------------------    
    with tab8:

        st.subheader("🔎 Jugadores similares")

        eligible_players = players[
            players["Minutos jugados"] >= min_minutes
        ]["Jugador"].unique().tolist()

        selected_player = st.selectbox("Jugador", eligible_players)

        selected_role = st.selectbox("Rol", list(role_scores.keys()))

        if selected_player and selected_role:

            similar_df = find_similar_players(
                players,
                selected_role,
                selected_player,
                min_minutes,
                top_n=5
            )

            if not similar_df.empty:
                st.dataframe(similar_df, use_container_width=True)
            else:
                st.warning("No se encontraron similares.")