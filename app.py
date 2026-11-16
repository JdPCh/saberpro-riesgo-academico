"""Saber Pro Analytics · desktop Streamlit application."""
from html import escape
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from profile_model import (
    GROUPS, LABELS, RECOMMENDATIONS, display_category, predict_profile,
    risk_level, validate_profile,
)

ROOT = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/jem2002/saberpro-riesgo-academico"
RESPONSIBLE_USE = (
    " "
    ""
)
ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M4 19V5M4 19h16M8 15v-4m5 4V7m5 8v-6"/></svg>'


def decimal(value, places=1):
    return f"{value:.{places}f}".replace(".", ",")


def percent(value):
    return f"{decimal(value * 100)} %"


def integer(value):
    return f"{int(value):,}".replace(",", ".")


@st.cache_resource(show_spinner=False)
def load_artifact():
    return joblib.load(ROOT / "modelo_riesgo.joblib")


@st.cache_data(show_spinner=False)
def load_csv(name):
    return pd.read_csv(ROOT / "resultados" / name)


@st.cache_data(show_spinner=False)
def load_metrics():
    return json.loads((ROOT / "resultados" / "metricas_finales.json").read_text(encoding="utf-8"))


def section_heading(number, title):
    st.markdown(f'<h3 class="form-heading"><span>{number:02d}</span>{escape(title)}</h3>', unsafe_allow_html=True)


def calculate(artifact):
    """Runs after the native form has committed all widget values."""
    data = {column: st.session_state.get(f"profile_{column}") for column in LABELS}
    errors = validate_profile(data, artifact)
    st.session_state["profile_errors"] = errors
    st.session_state.pop("prediction_error", None)
    st.session_state.pop("prediction", None)
    if not errors:
        try:
            with st.spinner("Calculando la estimación…"):
                st.session_state["prediction"] = {
                    "probability": predict_profile(data, artifact), "profile": data.copy(),
                }
        except Exception:
            logging.exception("No se pudo calcular la estimación")
            st.session_state["prediction_error"] = "No pudimos calcular la estimación. Conservamos tus respuestas; vuelve a intentarlo."


def render_field(column, artifact):
    if column == "edad":
        st.number_input("Edad", min_value=16, max_value=70, value=None, step=1,
                        placeholder="16–70 años", key="profile_edad")
    else:
        st.selectbox(LABELS[column], options=artifact["opciones"][column], index=None,
                     placeholder="Selecciona…", format_func=display_category,
                     key=f"profile_{column}")
    error = st.session_state.get("profile_errors", {}).get(column)
    if error:
        st.markdown(f'<p class="field-error" role="alert">{escape(error)}</p>', unsafe_allow_html=True)


def render_scale(probability, cuts):
    low, high = (float(c) * 100 for c in cuts)
    marker = "" if probability is None else f'<span class="scale-marker" style="left:clamp(2px, {probability * 100:.6f}%, calc(100% - 2px))"></span>'
    description = (f"Escala de probabilidad. Bajo: menos de {low:g} %. "
                   f"Medio: desde {low:g} % hasta menos de {high:g} %. Alto: desde {high:g} %.")
    if probability is not None:
        description += f" Estimación: {percent(probability)}."
    st.markdown(f"""
<div class="risk-scale" role="img" aria-label="{escape(description)}">
<div class="scale-track"><span class="scale-low" style="width:{low}%"></span><span class="scale-medium" style="width:{high-low}%"></span><span class="scale-high" style="width:{100-high}%"></span>{marker}</div>
<div class="scale-ticks"><span>0 %</span><span style="left:{low}%">{low:g} %</span><span style="left:{high}%">{high:g} %</span><span>100 %</span></div>
<div class="scale-legend"><span><i class="dot-low"></i>Bajo</span><span><i class="dot-medium"></i>Medio</span><span><i class="dot-high"></i>Alto</span></div>
</div>""", unsafe_allow_html=True)


def render_result(artifact):
    with st.container(key="result_rail"):
        with st.container(border=True, key="result_card"):
            st.markdown("## Estimación de riesgo")
            result = st.session_state.get("prediction")
            if result is None:
                st.markdown(f"""<div class="empty-result"><div class="empty-icon">{ICON}</div>
<h3>Aún no hay una estimación</h3><p>Completa el formulario para conocer el riesgo académico estimado</p></div>""", unsafe_allow_html=True)
                render_scale(None, artifact["cortes"])
                st.caption("La escala muestra la probabilidad de pertenecer al 25 % de menor desempeño.")
            else:
                probability = result["probability"]
                level = risk_level(probability, artifact["cortes"])
                st.markdown(f"""<div class="prediction" role="status" aria-live="polite">
<p class="probability-label">Probabilidad estimada</p><div class="probability">{decimal(probability * 100)}<span> %</span></div>
<span class="risk-badge risk-{level.lower()}">Riesgo {level.lower()}</span></div>""", unsafe_allow_html=True)
                render_scale(probability, artifact["cortes"])
                base = float(artifact["base"])
                ratio = f"{decimal(probability / base)}×" if base > 0 else "No disponible"
                st.markdown(f"""<dl class="result-context"><div><dt>Riesgo promedio histórico</dt><dd>{percent(base)}</dd></div>
<div><dt>Frente al promedio</dt><dd>{ratio}</dd></div></dl>""", unsafe_allow_html=True)
                st.caption("Referencia: conjunto de entrenamiento. 1× equivale al riesgo promedio histórico.")
                st.markdown("### Orientación académica")
                st.write(RECOMMENDATIONS[level])
                st.caption("Orientación de apoyo; no constituye un diagnóstico individual. Si cambias respuestas, vuelve a calcular para actualizar la estimación.")
                with st.expander("Respuestas utilizadas"):
                    st.dataframe(pd.DataFrame([
                        {"Información": LABELS[c], "Respuesta": str(display_category(v)) if c != "edad" else f"{v} años"}
                        for c, v in result["profile"].items()
                    ]), hide_index=True, width="stretch")
        st.markdown(f'<p class="responsible-note">{escape(RESPONSIBLE_USE)}</p>', unsafe_allow_html=True)
        with st.expander("Sobre esta estimación"):
            st.markdown(f"**Modelo:** {artifact['modelo_nombre']}  \n**ROC-AUC en prueba:** {decimal(artifact['auc_test'], 3)}")
            st.caption("ROC-AUC mide la capacidad de distinguir perfiles con y sin riesgo. No es un porcentaje de aciertos.")
            st.write("El objetivo es pertenecer al 25 % inferior del puntaje compuesto de la muestra: promedio de razonamiento cuantitativo, lectura crítica, inglés y competencias ciudadanas.")
            st.caption("Los umbrales de la escala se conservan del modelo entrenado.")


def render_evaluation(artifact):
    left, right = st.columns([1.68, 1], gap="large")
    with left:
        st.markdown("## Evaluar un perfil")
        st.markdown('<p class="section-intro">Completa los cuatro grupos de información. Todos los campos requieren una respuesta.</p>', unsafe_allow_html=True)
        with st.container(border=True, key="input_panel"), st.form("profile_form", clear_on_submit=False, border=False):
            for number, (title, rows) in enumerate(GROUPS, start=1):
                section_heading(number, title)
                for row in rows:
                    for container, column in zip(st.columns(len(row), gap="small"), row):
                        with container:
                            render_field(column, artifact)
                if number < len(GROUPS):
                    st.divider()
            if st.session_state.get("profile_errors"):
                count = len(st.session_state["profile_errors"])
                st.error(f"Revisa {count} {'campo pendiente' if count == 1 else 'campos pendientes'}. Encontrarás una indicación junto a cada respuesta.")
            if st.session_state.get("prediction_error"):
                st.error(st.session_state["prediction_error"])
            st.form_submit_button("Calcular riesgo", type="primary", width="stretch", on_click=calculate, args=(artifact,))
            st.caption(RESPONSIBLE_USE)
    with right:
        render_result(artifact)


def chart(data, spec, height=240):
    st.vega_lite_chart(data, {
        "height": height,
        "config": {
            "view": {"stroke": None},
            "axis": {"labelColor": "#475569", "titleColor": "#475569", "gridColor": "#E2E8F0", "domain": False, "ticks": False, "labelFontSize": 12, "titleFontSize": 12},
            "legend": {"labelColor": "#475569", "title": None},
            "font": "Source Sans Pro, sans-serif",
        }, **spec,
    })


def render_findings(artifact):
    st.markdown("## Hallazgos del modelo")
    st.markdown('<p class="section-intro">Resultados del análisis y evaluación. Las asociaciones observadas no implican causalidad.</p>', unsafe_allow_html=True)
    try:
        metrics = load_metrics()
        comparison = load_csv("comparacion_modelos_cv.csv")
        importance = load_csv("importancia_variables.csv")
        calibration = load_csv("calibracion_por_tramos.csv")
        rates = load_csv("tasa_riesgo_por_categoria.csv")
    except (OSError, ValueError):
        logging.exception("No se pudieron cargar los resultados")
        st.warning("No pudimos cargar los resultados del análisis. La evaluación de perfiles sigue disponible.")
        return
    with st.container(border=True, key="finding_metrics"):
        for col, label, value in zip(st.columns(4),
            ["ROC-AUC · prueba", "PR-AUC · prueba", "Perfiles de prueba", "Riesgo base · entrenamiento"],
            [decimal(metrics["roc_auc_test"], 3), decimal(metrics["pr_auc_test"], 3), integer(metrics["n_test"]), percent(artifact["base"])],
        ):
            col.metric(label, value)
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="finding_comparison"):
            st.markdown("### Comparación de modelos")
            st.caption("Validación cruzada de 5 particiones en entrenamiento. Media ± desviación estándar para ROC-AUC y PR-AUC.")
            st.dataframe(comparison.rename(columns={"Modelo": "Modelo entrenado"}), hide_index=True, width="stretch")
            st.markdown(f"**Seleccionado:** {escape(artifact['modelo_nombre'])}.")
            st.caption("ROC-AUC evalúa discriminación; PR-AUC resume precisión y exhaustividad en la clase de riesgo. Ninguna equivale al porcentaje de aciertos.")
        with st.container(border=True, key="finding_confusion"):
            st.markdown("### Matriz de confusión")
            st.caption(f"Conjunto de prueba · umbral de clasificación: {decimal(metrics['umbral_clasificacion'], 3)}. Este umbral es distinto de los cortes bajo / medio / alto.")
            st.markdown(f"""<table class="confusion-table"><caption>Observado en filas · predicho en columnas</caption>
<thead><tr><th scope="col">Observado / predicho</th><th scope="col">Sin riesgo</th><th scope="col">Con riesgo</th></tr></thead>
<tbody><tr><th scope="row">Sin riesgo</th><td class="correct"><strong>{integer(metrics['TN'])}</strong><span>Verdaderos negativos</span></td><td><strong>{integer(metrics['FP'])}</strong><span>Falsos positivos</span></td></tr>
<tr><th scope="row">Con riesgo</th><td><strong>{integer(metrics['FN'])}</strong><span>Falsos negativos</span></td><td class="correct"><strong>{integer(metrics['TP'])}</strong><span>Verdaderos positivos</span></td></tr></tbody></table>""", unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric("Precisión · clase riesgo", decimal(metrics["precision_riesgo"], 3))
            b.metric("Recall · clase riesgo", decimal(metrics["recall_riesgo"], 3))
    with right:
        with st.container(border=True, key="finding_importance"):
            st.markdown("### Variables con mayor importancia")
            st.caption("Caída de ROC-AUC al permutar cada variable. Una caída mayor indica más importancia global.")
            ranked = importance.sort_values("caida_auc", ascending=False).head(8).copy()
            ranked["Variable"] = ranked["variable"].map(LABELS)
            chart(ranked, {
                "mark": {"type": "bar", "color": "#0F766E", "cornerRadiusEnd": 3},
                "encoding": {
                    "y": {"field": "Variable", "type": "nominal", "sort": "-x", "title": None, "axis": {"labelLimit": 250}},
                    "x": {"field": "caida_auc", "type": "quantitative", "title": "Caída de ROC-AUC"},
                    "tooltip": [{"field": "Variable"}, {"field": "caida_auc", "title": "Caída AUC", "format": ".4f"}, {"field": "desv", "title": "Desviación", "format": ".4f"}],
                },
            })
            st.caption("Describe el modelo en conjunto; no explica la predicción de una persona.")
            with st.expander("Ver las 16 variables"):
                table = importance.copy()
                table["variable"] = table["variable"].map(LABELS)
                st.dataframe(table.rename(columns={"variable": "Variable", "caida_auc": "Caída AUC", "desv": "Desviación"}), hide_index=True, width="stretch")
        with st.container(border=True, key="finding_calibration"):
            st.markdown("### Calibración de probabilidades")
            st.caption("Compara la probabilidad media estimada con la proporción observada de riesgo en los nueve tramos de prueba utilizados en el informe.")
            chart(calibration, {
                "layer": [
                    {"data": {"values": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}, "mark": {"type": "line", "color": "#64748B", "strokeDash": [4, 4]}, "encoding": {"x": {"field": "x", "type": "quantitative", "title": "Probabilidad estimada", "axis": {"format": ".0%"}}, "y": {"field": "y", "type": "quantitative", "title": "Riesgo observado", "axis": {"format": ".0%"}}}},
                    {"mark": {"type": "line", "point": True, "color": "#0F766E"}, "encoding": {
                        "x": {"field": "prob_media", "type": "quantitative", "title": "Probabilidad estimada", "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}},
                        "y": {"field": "riesgo_real", "type": "quantitative", "title": "Riesgo observado", "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}},
                        "tooltip": [{"field": "tramo", "title": "Tramo"}, {"field": "n", "title": "Perfiles"}, {"field": "prob_media", "title": "Estimada", "format": ".1%"}, {"field": "riesgo_real", "title": "Observada", "format": ".1%"}],
                    }},
                ],
            }, height=210)
            st.caption("Línea discontinua: calibración ideal. Línea verde: resultados observados.")
    with st.container(border=True, key="finding_subgroups"):
        st.markdown("### Diferencias observadas entre perfiles")
        selected = st.selectbox("Comparar por", rates["variable"].unique().tolist(), format_func=lambda c: LABELS.get(c, c), key="finding_group")
        subset = rates.loc[rates["variable"] == selected].copy()
        subset["Categoría"] = subset["categoria"].map(display_category)
        chart(subset, {
            "mark": {"type": "bar", "color": "#142C45", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": "Categoría", "type": "nominal", "title": None, "sort": None, "axis": {"labelLimit": 300}},
                "x": {"field": "tasa_riesgo", "type": "quantitative", "title": "Proporción observada de riesgo", "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}},
                "tooltip": [{"field": "Categoría"}, {"field": "n", "title": "Perfiles"}, {"field": "tasa_riesgo", "title": "Riesgo observado", "format": ".1%"}],
            },
        }, height=max(150, len(subset) * 32))
        st.caption("Datos históricos de la muestra; no son efectos causales ni pronósticos individuales. Los grupos con pocos casos requieren cautela.")
        with st.expander("Ver valores y tamaño de cada grupo"):
            st.dataframe(subset[["Categoría", "n", "tasa_riesgo"]].rename(columns={"n": "Perfiles", "tasa_riesgo": "Proporción de riesgo"}), hide_index=True, width="stretch")
    with st.expander("Figuras originales del análisis"):
        st.caption("Figuras guardadas en el repositorio, sin modificar sus resultados.")
        a, b = st.columns(2)
        for col, filename, title in [(a, "evaluacion.png", "Evaluación del modelo"), (b, "calibracion.png", "Calibración"), (a, "importancias.png", "Importancia global"), (b, "eda_puntaje.png", "Distribución del puntaje")]:
            with col:
                path = ROOT / "figs" / filename
                if path.is_file():
                    st.markdown(f"#### {title}")
                    st.image(str(path), width="stretch")
                    if filename == "calibracion.png":
                        st.caption("Esta figura original usa 10 grupos por cuantiles. El gráfico principal de calibración usa los nueve tramos del CSV, como el informe.")



def render_about(artifact):
    st.markdown("## Acerca del proyecto")
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        with st.container(border=True, key="about_context"):
            st.markdown("### Apoyo al acompañamiento académico")
            st.write("El sistema estima la probabilidad de pertenecer al 25 % inferior de un puntaje compuesto de Saber Pro a partir de información personal, del hogar, del programa y de la institución. Su propósito es orientar el seguimiento y las necesidades de apoyo académico.")
            st.markdown("### Datos utilizados")
            st.write("La muestra contiene 154.984 registros de seis periodos entre 2018 y 2022 (20183, 20195, 20203, 20212, 20222 y 20225), extraídos mediante muestreo estratificado por periodo con asignación proporcional.")
            st.markdown("Fuente: [Resultados únicos Saber Pro · Datos Abiertos Colombia](https://www.datos.gov.co/resource/u37r-hjmu).")
            st.write("El puntaje compuesto es el promedio de razonamiento cuantitativo, lectura crítica, inglés y competencias ciudadanas. Es una medida construida para el proyecto; no equivale al puntaje global oficial del ICFES.")
            st.markdown("### Metodología CRISP-DM")
            st.markdown("1. **Comprensión del problema:** definir el riesgo y el objetivo de acompañamiento.\n2. **Comprensión de los datos:** revisar calidad y representatividad.\n3. **Preparación:** derivar la edad, imputar y codificar dentro del pipeline.\n4. **Modelado:** comparar cuatro algoritmos con validación cruzada.\n5. **Evaluación:** revisar discriminación, calibración y subgrupos.\n6. **Despliegue:** ofrecer la estimación en Streamlit.")
    with right:
        with st.container(border=True, key="about_limits"):
            st.markdown("### Modelo seleccionado")
            st.write(f"**{artifact['modelo_nombre']}**")
            st.write("HistGradientBoostingClassifier de scikit-learn, con el preprocesamiento integrado en el pipeline entrenado.")
            st.markdown("### Limitaciones")
            st.markdown("- El riesgo es relativo a la muestra histórica, no una medida absoluta de desempeño.\n- Los datos de 2018–2022 pueden representar de forma distinta a estudiantes de otros periodos.\n- Las variables muestran asociaciones, no relaciones de causa y efecto.\n- La estimación tiene incertidumbre y necesita contexto académico individual.")
        st.markdown(f'<p class="responsible-note">{escape(RESPONSIBLE_USE)}</p>', unsafe_allow_html=True)
        st.link_button("Consultar el repositorio", REPOSITORY, width="stretch")
        st.caption("Joshua Esteban Martínez Carvajal  \nCristian Camilo Soto Robles  \nJesús David Peña Chivatá")


def render_workspace():
    st.markdown(f"""<header class="app-header"><div class="brand-icon">{ICON}</div><div>
<h1>Saber Pro Analytics</h1><p>Estimación del riesgo académico mediante aprendizaje automático</p></div>
</header>
<p class="app-description">Estima la probabilidad de bajo desempeño en Saber Pro a partir de información personal, socioeconómica e institucional.</p>""", unsafe_allow_html=True)
    try:
        artifact = load_artifact()
    except Exception:
        logging.exception("No se pudo cargar el artefacto del modelo")
        st.error("No pudimos cargar el modelo. Vuelve a abrir la aplicación o informa al responsable del proyecto.")
        st.stop()
    evaluation, findings, about = st.tabs(["Evaluar perfil", "Hallazgos del modelo", "Acerca del proyecto"])
    with evaluation:
        render_evaluation(artifact)
    with findings:
        render_findings(artifact)
    with about:
        render_about(artifact)



def main():
    st.set_page_config(page_title="Saber Pro Analytics", page_icon=":material/analytics:", layout="wide")
    st.markdown(f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.container(key="workspace"):
        render_workspace()


if __name__ == "__main__":
    main()
