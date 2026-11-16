"""Presentation metadata and the unchanged inference contract of the artifact."""
import math

import pandas as pd

LABELS = {
    "estu_genero": "Género", "estu_depto_reside": "Departamento de residencia", "edad": "Edad",
    "fami_estratovivienda": "Estrato de la vivienda", "fami_educacionpadre": "Educación del padre",
    "fami_educacionmadre": "Educación de la madre", "fami_tieneinternet": "Acceso a internet",
    "fami_tienecomputador": "Computador", "fami_tieneautomovil": "Automóvil", "fami_tienelavadora": "Lavadora",
    "estu_horassemanatrabaja": "Horas semanales de trabajo", "estu_valormatriculauniversidad": "Valor de la matrícula",
    "estu_metodo_prgm": "Modalidad del programa", "inst_caracter_academico": "Carácter académico de la institución",
    "inst_origen": "Origen de la institución", "estu_nucleo_pregrado": "Núcleo del pregrado",
}

GROUPS = (
    ("Información personal", (("estu_genero", "edad"), ("estu_depto_reside",))),
    ("Condiciones del hogar", (
        ("fami_estratovivienda",),
        ("fami_educacionpadre", "fami_educacionmadre"),
        ("fami_tieneinternet", "fami_tienecomputador"),
        ("fami_tieneautomovil", "fami_tienelavadora"),
    )),
    ("Estudio y trabajo", (("estu_horassemanatrabaja", "estu_valormatriculauniversidad"),)),
    ("Programa e institución", (
        ("estu_metodo_prgm", "inst_origen"),
        ("inst_caracter_academico",), ("estu_nucleo_pregrado",),
    )),
)

RECOMMENDATIONS = {
    "Bajo": "El perfil presenta una probabilidad relativamente baja de pertenecer al grupo de menor desempeño. Se recomienda mantener hábitos de estudio y seguimiento académico.",
    "Medio": "El perfil presenta un nivel intermedio de riesgo. Se recomienda revisar posibles necesidades de refuerzo y realizar seguimiento académico.",
    "Alto": "El perfil presenta una probabilidad elevada de pertenecer al grupo de menor desempeño. Se recomienda priorizar una evaluación de necesidades de acompañamiento, tutorías y apoyo académico.",
}


def display_category(value):
    """Format the visible label without changing the value sent to the model."""
    text = str(value)
    aliases = {"F": "Femenino", "M": "Masculino", "Si": "Sí", "SI": "Sí",
               "DISTANCIA VITUAL": "Distancia virtual", "0": "No trabaja"}
    return aliases.get(text, text.capitalize() if text.isupper() else text)


def validate_profile(data, artifact):
    errors = {}
    for column in artifact["cat"]:
        value = data.get(column)
        if value is None:
            errors[column] = "Selecciona una opción."
        elif value not in artifact["opciones"][column]:
            errors[column] = "Selecciona una de las opciones disponibles."
    age = data.get("edad")
    if age is None:
        errors["edad"] = "Ingresa la edad."
    elif not isinstance(age, (int, float)) or not math.isfinite(age) or not 16 <= age <= 70 or int(age) != age:
        errors["edad"] = "Ingresa una edad entera entre 16 y 70 años."
    return errors


def risk_level(probability, cuts):
    low, high = cuts
    return "Alto" if probability >= high else "Medio" if probability >= low else "Bajo"


def predict_profile(data, artifact):
    if validate_profile(data, artifact):
        raise ValueError("El perfil está incompleto o contiene respuestas no válidas.")
    frame = pd.DataFrame([data], columns=[*artifact["cat"], *artifact["num"]])
    return float(artifact["modelo"].predict_proba(frame)[:, 1][0])
