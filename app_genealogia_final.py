import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Cargar datos
@st.cache_data
def load_data():
    df_gen = pd.read_csv('resultados_genealogia_actualizados.csv', dtype=str)
    df_ird = pd.read_csv('ird_por_caballo.csv', dtype=str)
    df_ird['IRD'] = pd.to_numeric(df_ird['IRD'], errors='coerce')
    df_irr = pd.read_csv('irr_por_caballo_final.csv', dtype=str)
    return df_gen, df_ird, df_irr

df_gen, df_ird, df_irr = load_data()

# Historial de navegación
if 'history' not in st.session_state:
    st.session_state.history = []

# Función de búsqueda por registro o nombre+criadero
def search_horse(query):
    q = query.lower().strip()
    if q.isdigit():
        return df_gen[df_gen['registro'] == q]
    return df_gen[
        df_gen['nombre'].str.lower().str.contains(q) &
        df_gen['criadero'].str.lower().str.contains(q)
    ]

# Extraer roles genealógicos
@st.cache_data
def extract_roles(df):
    df = df.copy()
    df['registro_padre'] = df['padres'].str.split('|').str[0].str.split(':').str[0]
    df['registro_madre'] = df['padres'].str.split('|').str[1].str.split(':').str[0]
    def extract_ab(reg_str, idx):
        parts = reg_str.split('|') if pd.notna(reg_str) else []
        regs = [p.split(':')[0] for p in parts]
        regs += [None] * (4 - len(regs))
        return regs[idx]
    cols = [
        'registro_abuelo_paterno', 'registro_abuela_paterna',
        'registro_abuelo_materno', 'registro_abuela_materna'
    ]
    for i, col in enumerate(cols):
        df[col] = df['abuelos'].apply(lambda x, i=i: extract_ab(x, i))
    return df

df_gen = extract_roles(df_gen)

# Mostrar árbol genealógico
def show_tree(row):
    G = nx.DiGraph()
    label = f"{row['nombre']}\n({row['criadero']})"
    G.add_node(label)
    for col in ['registro_padre', 'registro_madre']:
        anc = row.get(col)
        if pd.notna(anc):
            G.add_node(anc)
            G.add_edge(anc, label)
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, ax=ax)
    st.pyplot(fig)

# Mostrar ficha del caballo
def show_profile(row):
    st.header(f"{row['nombre']} - {row['criadero']} (Reg: {row['registro']})")
    st.text(
        f"Nacimiento: {row.get('nacimiento','N/A')}    "
        f"Sexo: {row.get('sexo','N/A')}    "
        f"Color: {row.get('color','N/A')}"
    )
    # IRD y percentil en su año
    ird_val = df_ird[df_ird['registro'] == row['registro']]['IRD'].astype(float)
    if not ird_val.empty:
        year = row.get('nacimiento')
        merged = df_ird.merge(df_gen[['registro','nacimiento']], on='registro', how='left')
        same_year = merged[merged['nacimiento'] == year]['IRD'].dropna().astype(float)
        if not same_year.empty:
            pctl = (same_year < ird_val.iloc[0]).mean() * 100
            st.metric('IRD', f"{ird_val.iloc[0]:.2f}", delta=f"Percentil: {pctl:.1f}%")
        else:
            st.metric('IRD', f"{ird_val.iloc[0]:.2f}")
    # IRR por rol
    st.subheader('IRR por rol')
    irr_row = df_irr[df_irr['registro'] == row['registro']]
    if not irr_row.empty:
        for role in ['padre','madre','abuelo_paterno','abuela_paterna','abuelo_materno','abuela_materna']:
            val = irr_row.iloc[0].get(f"{role}_irr")
            if pd.notna(val):
                st.write(f"- {role.replace('_',' ').title()}: {float(val):.3f}")
    # Árbol genealógico
    st.subheader('Árbol Genealógico')
    show_tree(row)
    # Descendientes
    st.subheader('Descendientes')
    children = pd.concat([
        df_gen[df_gen['registro_padre'] == row['registro']],
        df_gen[df_gen['registro_madre'] == row['registro']]
    ])
    if not children.empty:
        df_h = children[['registro','nombre','criadero','nacimiento','sexo']].copy()
        df_h = df_h.merge(df_ird[['registro','IRD']], on='registro', how='left')
        df_h = df_h.merge(
            df_irr[['registro'] + [f"{r}_irr" for r in ['padre','madre','abuelo_paterno','abuela_paterna','abuelo_materno','abuela_materna']]],
            on='registro', how='left'
        )
        sex = st.selectbox('Filtrar por sexo',['Todos','Macho','Hembra'], key='sex')
        has_ird = st.selectbox('Filtrar IRD',['Todos','Con IRD','Sin IRD'], key='has_ird')
