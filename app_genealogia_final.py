
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df_gen = pd.read_csv('resultados_genealogia_actualizados.csv', dtype=str)
    df_ird = pd.read_csv('ird_por_caballo.csv', dtype=str)
    df_ird['IRD'] = pd.to_numeric(df_ird['IRD'], errors='coerce')
    df_irr = pd.read_csv('irr_por_caballo_final.csv', dtype=str)
    return df_gen, df_ird, df_irr

df_gen, df_ird, df_irr = load_data()

# Session history
if 'history' not in st.session_state:
    st.session_state.history = []

# Search
def search_horse(q):
    q = q.lower().strip()
    if q.isdigit():
        return df_gen[df_gen['registro']==q]
    return df_gen[df_gen['nombre'].str.lower().str.contains(q) & df_gen['criadero'].str.lower().str.contains(q)]

# Display pedigree tree
def show_tree(row):
    G = nx.DiGraph()
    label = f"{row['nombre']}\n({row['criadero']})"
    G.add_node(label)
    for r, role in [('registro_padre','Padre'),('registro_madre','Madre')]:
        anc = row.get(r)
        if pd.notna(anc):
            G.add_node(anc)
            G.add_edge(anc, label)
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, ax=ax)
    st.pyplot(fig)

# Show horse profile
def show_profile(row):
        st.header(
        f"{row['nombre']} - {row['criadero']} (Reg: {row['registro']})"
    )
    st.text(
        f"Nacimiento: {row.get('nacimiento','N/A')}    "
        f"Sexo: {row.get('sexo','N/A')}    "
        f"Color: {row.get('color','N/A')}"
    )
   Color: {row.get('color','N/A')}")
    # IRD percentile
    ird_val = df_ird[df_ird['registro']==row['registro']]['IRD'].astype(float)
    if not ird_val.empty:
        same_year = df_ird[df_gen['nacimiento']==row['nacimiento']]['IRD'].astype(float)
        pctl = (ird_val.iloc[0] >= same_year.quantile([0.01*i for i in range(100)])).sum() 
        st.metric('IRD', f"{ird_val.iloc[0]:.2f}", delta=f"Percentil: {pctl}%")

    # IRR roles
    st.subheader('IRR por rol')
    irr_row = df_irr[df_irr['registro']==row['registro']]
    if not irr_row.empty:
        for role in ['padre', 'madre', 'abuelo_paterno', 'abuela_paterna', 'abuelo_materno', 'abuela_materna']:
            val = irr_row.iloc[0].get(f"{role}_irr")
            if pd.notna(val):
                st.write(f"- {role.replace('_',' ').title()}: {float(val):.3f}")

    # Pedigree
    st.subheader('Árbol Genealógico')
    show_tree(row)

    # Descendants
    st.subheader('Descendientes')
    desc = pd.concat([
        df_gen[df_gen['padres'].str.split('|').str[0].str.split(':').str[0]==row['registro']],
        df_gen[df_gen['padres'].str.split('|').str[1].str.split(':').str[0]==row['registro']]
    ])
    if not desc.empty:
        cols = ['nombre','criadero','registro']
        df_h = desc[cols].copy()
        df_h = df_h.merge(df_ird[['registro','IRD']], on='registro', how='left')
        df_h = df_h.merge(df_irr[['registro']+ [f"{r}_irr" for r in roles]], on='registro', how='left')
        # Filters & ordering
        sex = st.selectbox('Sexo',['Todos','Macho','Hembra'], key='sex')
        has_ird = st.selectbox('Con IRD',['Todos','Con IRD','Sin IRD'], key='has_ird')
        order = st.selectbox('Orden', ['Nombre','IRD'], key='order')
        if sex!='Todos':
            df_h = df_h[df_gen['sexo']==sex]
        if has_ird!='Todos':
            df_h = df_h[df_h['IRD'].notna()] if has_ird=='Con IRD' else df_h[df_h['IRD'].isna()]
        if order=='Nombre':
            df_h = df_h.sort_values('nombre')
        else:
            df_h = df_h.sort_values('IRD', ascending=False)
        for idx, hr in df_h.iterrows():
            if st.button(f"Perfil: {hr['nombre']} ({hr['registro']})"):
                st.session_state.history.append(row['registro'])
                show_profile(hr)

# Main
st.title('Genealogía Caballos Chilenos')
q = st.text_input('Buscar registro o nombre/criadero')
if q:
    res = search_horse(q)
    if res.empty:
        st.warning('No encontrado')
    else:
        idx = res.index[0]
        show_profile(res.loc[idx])
        if st.session_state.history:
            if st.button('Volver'):
                prev = st.session_state.history.pop()
                show_profile(df_gen[df_gen['registro']==prev].iloc[0])
else:
    st.info('Usa la búsqueda para encontrar un caballo.')
