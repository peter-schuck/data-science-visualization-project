import os
import re
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import rasterio
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt # <-- NOVA IMPORTAÇÃO
from rasterio.merge import merge
from rasterio.transform import rowcol

st.set_page_config(layout="wide", page_title="Enchente RS 2024 - Monitoramento Oficial")

# ─────────────────────────────────────────────────────────────
# 1. LEITURA E PROCESSAMENTO DE DADOS (HIDROGRAFIA E CLIMA)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_river_data():
    def parse_txt(file_name):
        try:
            with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
        except FileNotFoundError:
            return pd.DataFrame()

        pattern = r"<td>(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})</td>.*?<td[^>]*>(.*?)</td>.*?<td[^>]*>(.*?)</td>"
        rows = []
        for m in re.findall(pattern, html, re.DOTALL):
            val_chuva = m[1].strip().replace(',', '.')
            val_nivel = m[2].strip().replace(',', '.')
            if not val_nivel: continue
            try:
                rows.append({
                    "DataHora": pd.to_datetime(m[0].strip(), format="%d/%m/%Y %H:%M:%S"),
                    "Chuva_mm": float(val_chuva) if val_chuva else 0.0,
                    "Nivel_m": float(val_nivel) / 100.0,
                })
            except ValueError:
                continue
        return pd.DataFrame(rows).sort_values("DataHora").reset_index(drop=True)

    df_cais = parse_txt("cais_maua_c6.txt")
    df_gas  = parse_txt("usina_gasometro.txt")

    if df_cais.empty and df_gas.empty:
        st.error("Arquivos de nível não encontrados!")
        return pd.DataFrame()

    merged = pd.merge(df_cais.rename(columns={"Nivel_m": "Nivel_Cais", "Chuva_mm": "Chuva_Cais"}),
                      df_gas.rename(columns={"Nivel_m": "Nivel_Gas", "Chuva_mm": "Chuva_Gas"}),
                      on="DataHora", how="outer").sort_values("DataHora")
    
    merged["Nivel"] = merged["Nivel_Cais"].fillna(merged["Nivel_Gas"])
    merged["Chuva_mm"] = merged["Chuva_Gas"].fillna(merged["Chuva_Cais"]).fillna(0.0)

    df_timeline = merged[["DataHora", "Nivel", "Chuva_mm"]].set_index("DataHora").resample("30min").max().interpolate(method="linear").reset_index().dropna()
    df_timeline = df_timeline[df_timeline["DataHora"] <= "2024-07-04 23:59:59"]
    return df_timeline

# --- NOVA FUNÇÃO: CARREGAR DADOS DO INMET ---
@st.cache_data
def load_inmet_data():
    def carregar(file, bairro_nome):
        try:
            df = pd.read_csv(file, sep=';', decimal=',', encoding='latin1', skiprows=8, header=0)
            df.columns = ["data", "hora", "precipitacao", "pressao", "pressao_max", "pressao_min", "radiacao",
                          "temperatura", "ponto_orvalho", "temp_max", "temp_min", "orvalho_max", "orvalho_min",
                          "umidade_max", "umidade_min", "umidade", "vento_direcao", "vento_rajada", "vento_velocidade", "extra"]
            df['hora'] = df['hora'].astype(str).str.replace(' UTC', '', regex=False)
            df['datetime'] = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%Y/%m/%d %H%M')
            df = df.drop(columns=['data', 'hora', 'extra']).dropna(how='all')
            df['bairro'] = bairro_nome
            return df
        except Exception:
            return pd.DataFrame()

    df_bn = carregar("dados_climaticos_belem_novo.csv", "Belém Novo")
    df_jb = carregar("dados_climaticos_jardim_botanico.csv", "Jardim Botânico")
    
    if df_bn.empty and df_jb.empty:
        return pd.DataFrame()

    df_clima = pd.concat([df_bn, df_jb], ignore_index=True)
    df_daily = df_clima.resample('D', on='datetime').agg({
        'precipitacao': 'sum', 'temperatura': 'mean', 'temp_max': 'max', 
        'temp_min': 'min', 'umidade': 'mean', 'vento_velocidade': 'max'
    }).reset_index()
    
    df_daily['mes'] = df_daily['datetime'].dt.month
    return df_daily[df_daily['mes'] >= 4]

@st.cache_data
def load_tributary_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        pattern = r"<td>(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})</td>.*?<td[^>]*>(.*?)</td>.*?<td[^>]*>(.*?)</td>"
        rows = []
        for m in re.findall(pattern, html, re.DOTALL):
            val_nivel = m[2].strip().replace(',', '.')
            if not val_nivel: continue 
            try:
                rows.append({
                    "DataHora": pd.to_datetime(m[0].strip(), format="%d/%m/%Y %H:%M:%S"),
                    "Nivel_m": float(val_nivel) / 100.0 
                })
            except ValueError: continue
        return pd.DataFrame(rows).sort_values("DataHora").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def get_nivel_no_tempo(df, tempo_alvo):
    if df is None or df.empty: return None, None
    df_cortado = df[df["DataHora"] <= pd.to_datetime(tempo_alvo)]
    if len(df_cortado) < 2: return None, None
    ultimo, penultimo = df_cortado.iloc[-1], df_cortado.iloc[-2]
    return float(ultimo['Nivel_m']), float(ultimo['Nivel_m']) - float(penultimo['Nivel_m'])

# ─────────────────────────────────────────────────────────────
# 2. PROCESSAMENTO GEOESPACIAL (MAPAS)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_underlay_geojson(filepath):
    try:
        gdf = gpd.read_file(filepath)
        gdf['geometry'] = gdf['geometry'].simplify(0.0001, preserve_topology=True)
        return json.loads(gdf.to_json())
    except Exception: return None

@st.cache_data
def pre_compute_terrain(tif_paths: tuple, cell_size_deg: float = 0.005):
    srcs = [rasterio.open(fp) for fp in tif_paths if os.path.exists(fp)]
    if not srcs: return None, None, None, None
    elevation, transform = merge(srcs)
    elevation = elevation[0].astype(np.float32)
    for s in srcs: s.close()

    lon_min, lon_max, lat_min, lat_max = -51.8, -50.6, -30.8, -29.4
    lons = np.arange(lon_min, lon_max, cell_size_deg) + cell_size_deg / 2
    lats = np.arange(lat_min, lat_max, cell_size_deg) + cell_size_deg / 2
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat, lat_flat = lon_grid.ravel(), lat_grid.ravel()
    rows, cols = rowcol(transform, lon_flat, lat_flat)
    
    valid = ((rows >= 0) & (rows < elevation.shape[0]) & (cols >= 0) & (cols < elevation.shape[1]))
    alt, lons_v, lats_v = elevation[rows[valid], cols[valid]].astype(float), lon_flat[valid], lat_flat[valid]
    
    good = (alt > -9999) & (alt < 200)
    fator_lat = (lats_v[good] - lat_min) / (lat_max - lat_min)
    return lons_v[good], lats_v[good], alt[good], fator_lat

def calculate_flood_grid(lons, lats, alt, fator_lat, nivel: float, declividade: float):
    if lons is None: return pd.DataFrame()
    nivel_inclinado = nivel + (fator_lat * declividade)
    depth = np.maximum(0.0, nivel_inclinado - alt)
    flooded = depth > 0.1 
    if not flooded.any(): return pd.DataFrame()
    lons_f, lats_f, depth_f = lons[flooded], lats[flooded], depth[flooded]
    return pd.DataFrame({
        "lon": np.round(lons_f, 5), "lat": np.round(lats_f, 5),
        "height_3d": np.round(depth_f * 300, 1), 
        "r": np.select([depth_f < 1, depth_f < 2.5, depth_f < 4], [255, 255, 200], 100),
        "g": np.select([depth_f < 1, depth_f < 2.5, depth_f < 4], [200, 100, 0], 0),
        "b": 0, "a": np.select([depth_f < 1, depth_f < 2.5, depth_f < 4], [180, 200, 220], 240),
        "info": [f"Profundidade: {d:.2f} m" for d in depth_f],
    })

# ─────────────────────────────────────────────────────────────
# 3. INTERFACE DO USUÁRIO
# ─────────────────────────────────────────────────────────────
df_niveis = load_river_data()
df_inmet = load_inmet_data() # Carregamento dos dados climáticos

min_time, max_time = df_niveis["DataHora"].min().to_pydatetime(), df_niveis["DataHora"].max().to_pydatetime()
momento_pico, nivel_pico = df_niveis.loc[df_niveis['Nivel'].idxmax(), 'DataHora'].to_pydatetime(), df_niveis['Nivel'].max()
base_lons, base_lats, base_alt, base_fator_lat = pre_compute_terrain(("dem_final_vaga.tif",))
geojson_data = load_underlay_geojson("mancha_rm.geojson")

st.sidebar.header("Controle Temporal")
if "tempo_slider" not in st.session_state: st.session_state.tempo_slider = datetime(2024, 5, 5, 12, 0)

def avancar_tempo(): st.session_state.tempo_slider = min(st.session_state.tempo_slider + timedelta(hours=3), max_time)
def voltar_tempo(): st.session_state.tempo_slider = max(st.session_state.tempo_slider - timedelta(hours=3), min_time)

c1, c2, c3 = st.sidebar.columns([1, 1, 2])
c1.button("<", on_click=voltar_tempo, use_container_width=True)
c2.button(">", on_click=avancar_tempo, use_container_width=True)
c3.button("Recorde", on_click=lambda: setattr(st.session_state, 'tempo_slider', momento_pico), use_container_width=True)

tempo_sel = st.sidebar.slider("Linha do tempo:", min_value=min_time, max_value=max_time, step=timedelta(minutes=30), format="DD/MM/YY - HH:mm", key="tempo_slider")
st.sidebar.markdown("---")
ativar_3d = st.sidebar.toggle("Ativar Cubos Topográficos", value=False)
declividade_rio = st.sidebar.slider("Declividade (m):", 0.0, 5.0, 2.5, 0.5)

# Métricas Principais
idx = (df_niveis["DataHora"] - pd.to_datetime(tempo_sel)).abs().idxmin()
nivel_atual = float(df_niveis.loc[idx, "Nivel"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Horário", tempo_sel.strftime("%d/%m/%Y %H:%M"))
m2.metric("Nível Guaíba", f"{nivel_atual:.2f} m", f"{nivel_atual - 3.0:+.2f} m")
m3.metric("Recorde", f"{nivel_pico:.2f} m", momento_pico.strftime('%d/%m %H:%M'), delta_color="off")
m4.metric("Situação", "🔴 ALARME" if nivel_atual > 5 else ("🟠 INUNDAÇÃO" if nivel_atual > 3 else "🟢 ESTÁVEL"))

st.markdown("### Bacias Contribuintes")
cols = st.columns(4)
rios = [("taquari.xls", "Taquari"), ("cai.xls", "Caí"), ("sinos.xls", "Sinos"), ("jacui.xls", "Jacuí")]
for i, (file, name) in enumerate(rios):
    df_r = load_tributary_data(file)
    n, d = get_nivel_no_tempo(df_r, tempo_sel)
    if n: cols[i].metric(f"Rio {name}", f"{n:.2f} m", f"{d:+.2f} m", delta_color="inverse")

# --- GRÁFICO 1: NÍVEL DO RIO E MARCAÇÕES DE INUNDAÇÃO (PLOTLY) ---
with st.expander("📊 Gráfico de Precipitação e Nível do Guaíba", expanded=False):
    df_d = df_niveis.set_index('DataHora').resample('D').agg({'Nivel': 'max', 'Chuva_mm': 'sum'}).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_d['DataHora'], y=df_d['Chuva_mm'], name="Chuva (mm)", marker_color='rgba(100,150,255,0.6)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_d['DataHora'], y=df_d['Nivel'], name="Nível (m)", line=dict(color='red', width=2)), secondary_y=True)
    
    # Adicionando Linhas de Referência (Cota e Recorde)
    fig.add_hline(y=3.00, line_dash="dash", line_color="orange", secondary_y=True)
    fig.add_annotation(x=0.01, y=3.00, xref="paper", yref="y2", text="Cota de Inundação (3.0m)", showarrow=False, font=dict(color="orange"), xanchor="left", yanchor="bottom", yshift=5)
    
    fig.add_hline(y=4.76, line_dash="dot", line_color="purple", secondary_y=True)
    fig.add_annotation(x=0.01, y=4.76, xref="paper", yref="y2", text="Recorde de 1941 (4.76m)", showarrow=False, font=dict(color="purple"), xanchor="left", yanchor="bottom", yshift=5)

    fig.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# --- GRÁFICO 2: PAINEL METEOROLÓGICO COMPLETO (ALTAIR) ---
with st.expander("🌤️ Painel Climático Detalhado (INMET)", expanded=False):
    if not df_inmet.empty:
        seletor = alt.selection_point(name='seletor', fields=['mes'], bind=alt.binding_range(min=4, max=12, step=1, name='Mês:'), value=5)
        base = alt.Chart(df_inmet).properties(height=120)
        eixo_oculto = alt.Axis(labels=False, ticks=False, title='')
        tt_data = alt.Tooltip('datetime:T', title='Data', format='%d/%m/%Y')
        
        chuva = base.mark_bar(color='#1f77b4').encode(
            x=alt.X('datetime:T', axis=eixo_oculto), y=alt.Y('precipitacao:Q', title='Chuva (mm)'),
            tooltip=[tt_data, alt.Tooltip('precipitacao:Q', title='Chuva (mm)', format='.1f')]
        ).properties(title='Chuva, Temperatura, Umidade e Vento (A partir de Abril/2024)')

        temp = base.mark_line(color='#d62728').encode(
            x=alt.X('datetime:T', axis=eixo_oculto), y=alt.Y('temperatura:Q', title='Temp (°C)', scale=alt.Scale(domain=[0, 40])),
            tooltip=[tt_data, alt.Tooltip('temp_max:Q', title='Máx'), alt.Tooltip('temperatura:Q', title='Méd'), alt.Tooltip('temp_min:Q', title='Mín')]
        )
        f_temp = base.mark_area(opacity=0.2, color='#d62728').encode(x=alt.X('datetime:T'), y=alt.Y('temp_max:Q'), y2=alt.Y2('temp_min:Q'))
        
        umid = base.mark_area(color='#2ca02c', opacity=0.5).encode(
            x=alt.X('datetime:T', axis=eixo_oculto), y=alt.Y('umidade:Q', title='Umidade (%)', scale=alt.Scale(domain=[0, 100])),
            tooltip=[tt_data, alt.Tooltip('umidade:Q', title='Umidade (%)', format='.1f')]
        )

        vento = base.mark_line(color='#ff7f0e', strokeWidth=2).encode(
            x=alt.X('datetime:T', axis=alt.Axis(format='%d/%m', title='Dia/Mês', labelAngle=0)), 
            y=alt.Y('vento_velocidade:Q', title='Vento (m/s)'), tooltip=[tt_data, alt.Tooltip('vento_velocidade:Q', title='Vento Máx (m/s)', format='.1f')]
        )

        painel_climatico = alt.vconcat(chuva, f_temp + temp, umid, vento).resolve_scale(x='shared').add_params(seletor).transform_filter(seletor).configure_view(stroke=None)
        
        # O use_container_width do Streamlit garante que o Altair preencha o layout
        st.altair_chart(painel_climatico, use_container_width=True)
    else:
        st.warning("Dados do INMET (arquivos CSV) não foram encontrados na pasta.")

# ── SEÇÃO DO MAPA ──
@st.fragment
def render_map_section(nivel, declividade, mostrar_3d):
    st.subheader("Simulador Volumétrico e Mancha Oficial")
    layers = []
    if geojson_data:
        layers.append(pdk.Layer("GeoJsonLayer", data=geojson_data, pickable=False, stroked=False, filled=True, get_fill_color=[255, 0, 0, 100]))
    if mostrar_3d:
        df_f = calculate_flood_grid(base_lons, base_lats, base_alt, base_fator_lat, round(nivel, 2), declividade)
        if not df_f.empty:
            layers.append(pdk.Layer("GridCellLayer", data=df_f, get_position=["lon", "lat"], cell_size=400, get_elevation="height_3d", get_fill_color=["r", "g", "b", "a"], extruded=True, pickable=True))
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=-30.15, longitude=-51.18, zoom=10, pitch=45 if mostrar_3d else 0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    ), use_container_width=True)

render_map_section(nivel_atual, declividade_rio, ativar_3d)

# ── IMPACTO SOCIAL (FINAL DA PÁGINA) ──
st.markdown("---")
with st.container():
    st.subheader("Impacto Social: Vítimas por Município")
    # Dados corrigidos conforme PDF (Total Óbitos: 185)
    d_ob = {
        'AGUDO': 1, 'ALVORADA': 1, 'ARROIO DO MEIO': 1, 'BARROS CASSAL': 1, 'BENTO GONÇALVES': 11,
        'BOA VISTA DO SUL': 2, 'BOM PRINCÍPIO': 1, 'CACHOEIRINHA': 1, 'CANELA': 2, 'CANOAS': 31,
        'CAPELA DE SANTANA': 1, 'CAPITÃO': 2, 'CAXIAS DO SUL': 9, 'CHARQUEADAS': 1, 'CRUZEIRO DO SUL': 14,
        'ELDORADO DO SUL': 6, 'ENCANTADO': 1, 'ESTRELA': 2, 'FARROUPILHA': 1, 'FORQUETINHA': 2,
        'GENERAL CÂMARA': 1, 'GRAMADO': 7, 'GUAÍBA': 1, 'ITAARA': 2, 'LAJEADO': 2, 'MARQUES DE SOUZA': 1,
        'MONTENEGRO': 1, 'NOVA PETRÓPOLIS': 1, 'NOVO HAMBURGO': 1, 'PÂNTANO GRANDE': 1, 'PAVERAMA': 2,
        'PINHAL GRANDE': 2, 'PORTO ALEGRE': 5, 'PUTINGA': 1, 'RELVADO': 1, 'ROCA SALES': 14,
        'SALVADOR DO SUL': 2, 'SANTA CRUZ DO SUL': 2, 'SANTA MARIA': 5, 'SÃO JERÔNIMO': 1,
        'SÃO JOÃO DO POLÊSINE': 1, 'SÃO LEOPOLDO': 9, 'SÃO VENDELINO': 2, 'SEGREDO': 1,
        'SERAFINA CORRÊA': 2, 'SILVEIRA MARTINS': 1, 'SINIMBU': 3, 'SOBRADINHO': 1, 'TAQUARA': 2,
        'TAQUARI': 2, 'TEUTÔNIA': 2, 'TRAVESSEIRO': 1, 'TRÊS COROAS': 3, 'VALE DO SOL': 1,
        'VENÂNCIO AIRES': 5, 'VERANÓPOLIS': 5
    }
    # Dados Desaparecidos (Total: 23)
    d_de = {
        'AGUDO': 1, 'BENTO GONÇALVES': 4, 'CAXIAS DO SUL': 1, 'CRUZEIRO DO SUL': 4, 'ENCANTADO': 2,
        'ESTRELA': 1, 'LAJEADO': 3, 'MARQUES DE SOUZA': 1, 'POÇO DAS ANTAS': 1, 'PORTO ALEGRE': 1,
        'RELVADO': 1, 'ROCA SALES': 2, 'SÃO LEOPOLDO': 1
    }
    
    muns = sorted(list(set(list(d_ob.keys()) + list(d_de.keys()))))
    df_v = pd.DataFrame([{"Mun": m.title(), "Óbitos": d_ob.get(m, 0), "Desap": d_de.get(m, 0), "Tot": d_ob.get(m, 0) + d_de.get(m, 0)} for m in muns]).sort_values("Tot", ascending=True)
    
    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(y=df_v["Mun"], x=df_v["Óbitos"], name="Óbitos", orientation='h', marker_color='#0B5394', text=df_v["Óbitos"].replace(0,""), textposition='auto', textangle=0))
    fig_v.add_trace(go.Bar(y=df_v["Mun"], x=df_v["Desap"], name="Desap.", orientation='h', marker_color='#FFFFFF', text=df_v["Desap"].replace(0,""), textposition='auto', textangle=0))
    
    fig_v.update_layout(barmode='stack', bargap=0.2, height=1300, margin=dict(l=0,r=20,t=30,b=0), xaxis=dict(visible=False))
    st.plotly_chart(fig_v, use_container_width=True)
    
    st.markdown("<h4 style='text-align: center;'>Resumo Geral de Vítimas</h4>", unsafe_allow_html=True)
    c_v1, c_v2, c_v3 = st.columns(3)
    c_v1.metric("Óbitos", "185")
    c_v2.metric("Desaparecidos", "23")
    c_v3.metric("Total", "208")