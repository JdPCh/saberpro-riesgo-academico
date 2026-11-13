import streamlit as st, joblib, pandas as pd

st.set_page_config(page_title="Riesgo académico – Saber Pro", page_icon="🎓")
art = joblib.load("modelo_riesgo.joblib")

ETIQ = {"estu_genero": "Género", "estu_depto_reside": "Departamento de residencia",
        "fami_estratovivienda": "Estrato de la vivienda", "fami_educacionpadre": "Educación del padre",
        "fami_educacionmadre": "Educación de la madre", "fami_tieneinternet": "¿Tiene internet?",
        "fami_tienecomputador": "¿Tiene computador?", "fami_tieneautomovil": "¿Tiene automóvil?",
        "fami_tienelavadora": "¿Tiene lavadora?", "estu_horassemanatrabaja": "Horas semanales de trabajo",
        "estu_valormatriculauniversidad": "Valor de la matrícula", "estu_metodo_prgm": "Modalidad del programa",
        "inst_caracter_academico": "Carácter académico de la institución", "inst_origen": "Origen de la institución",
        "estu_nucleo_pregrado": "Núcleo del pregrado"}

st.title("🎓 Estimador de riesgo académico – Saber Pro")
st.caption("Estima la probabilidad de quedar en el 25 % inferior del puntaje compuesto a partir del contexto "
           "socioeconómico, institucional y del programa. Es una herramienta de apoyo, no un diagnóstico individual.")

with st.form("formulario"):
    c1, c2 = st.columns(2)
    datos = {}
    for i, c in enumerate(art["cat"]):
        datos[c] = (c1 if i % 2 == 0 else c2).selectbox(ETIQ.get(c, c), art["opciones"][c])
    datos["edad"] = c1.number_input("Edad", min_value=16, max_value=70, value=24)
    enviado = st.form_submit_button("Calcular riesgo")

if enviado:
    p = float(art["modelo"].predict_proba(pd.DataFrame([datos]))[:, 1][0])
    bajo, alto = art["cortes"]
    st.metric("Probabilidad estimada de riesgo", f"{p:.1%}",
              delta=f"{p / art['base']:.1f}× el riesgo promedio ({art['base']:.0%})", delta_color="off")
    st.progress(min(p, 1.0))
    if p >= alto:
        st.error("Riesgo ALTO: se recomienda priorizar acompañamiento académico (tutorías, nivelación, bienestar).")
    elif p >= bajo:
        st.warning("Riesgo MEDIO: conviene hacer seguimiento.")
    else:
        st.success("Riesgo BAJO.")
    st.caption(f"Modelo: {art['modelo_nombre']} · ROC-AUC en prueba: {art['auc_test']:.3f}. "
               "'Riesgo' = quedar en el 25 % inferior del puntaje compuesto de la muestra.")
