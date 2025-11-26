import streamlit as st
import pandas as pd
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


st.set_page_config(page_title="AttackScan-IDS Dashboard", layout="wide")

# Tabs
intro_tab, eda_tab, model_tab, predict_tab = st.tabs([
    "Introducción", "Análisis Exploratorio (EDA)", "Resultados del Modelo", "Predicción en línea"
])

with intro_tab:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("AttackScan-IDS: Dashboard de Detección de Intrusos en IoT")
        st.markdown(
            "<span style='font-size:22px; color:#d9534f;'><b>Protegiendo redes IoT contra amenazas modernas y ataques cibernéticos.</b></span>",
            unsafe_allow_html=True
        )
    with col2:
        st.image("https://www.noticias.ltda/wp-content/uploads/2019/03/ataques-ciberneticos-comunes.png", caption="Ciberataques en IoT", use_container_width=True)

    st.markdown("---")
    col_intro, col_obj = st.columns(2)
    with col_intro:
        st.subheader("Introducción")
        st.markdown("""
        El crecimiento de dispositivos IoT ha traído consigo nuevos retos de seguridad. Los ataques a redes inteligentes pueden comprometer la privacidad, disponibilidad y funcionamiento de sistemas críticos. Este proyecto propone un pipeline completo para la detección automática de intrusiones en tráfico IoT, combinando limpieza de datos, análisis exploratorio, preprocesamiento y modelos avanzados de machine learning.
        
        El sistema permite analizar el tráfico, identificar patrones maliciosos y predecir ataques en tiempo real, facilitando la protección de infraestructuras conectadas.
        """)
    with col_obj:
        st.subheader("Objetivos")
        st.markdown("""
        - Desarrollar un pipeline robusto para la detección de ataques en IoT.
        - Analizar y seleccionar las características más relevantes del tráfico.
        - Entrenar y evaluar modelos de clasificación multicategoría.
        - Visualizar resultados y métricas clave en un dashboard interactivo.
        - Permitir la predicción en línea a partir de nuevos datos JSON.
        """)

with eda_tab:
        st.header("Análisis Exploratorio de Datos (EDA)")
        st.markdown("""
        Explora el dataset interactivo y visualiza:
        - Distribución de clases
        - Distribución de variables
        - Correlación entre variables
        - Análisis temporal
        """)

        # Cargar el dataset Silver (puedes cambiar a Gold si prefieres)
        @st.cache_data(show_spinner=False)
        def load_data(path):
            return pd.read_csv(path)

        data_path = "../data/silver/data_merge_and_cleaned.csv"
        try:
            df = load_data(data_path)
            st.success(f"Dataset cargado: {data_path}")
        except Exception as e:
            st.error(f"No se pudo cargar el dataset: {e}")
            df = None

        if df is not None:
            @st.cache_data(show_spinner=False)
            def get_numeric_cols(df):
                return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            col1, col2 = st.columns(2)
            # 1. Distribución de clases
            @st.cache_data(show_spinner=False)
            def plot_class_distribution(df):
                fig, ax = plt.subplots(figsize=(5,3))
                df['Label'].value_counts().plot(kind='bar', ax=ax, color=['#5cb85c', '#d9534f'])
                ax.set_title("Distribución de clases")
                ax.set_xlabel("Clase (0 = Benign, 1 = Attack)")
                ax.set_ylabel("Cantidad")
                return fig
            with col1:
                st.subheader("Distribución de clases (Benign vs Attack)")
                st.pyplot(plot_class_distribution(df))

            # 2. Distribución de variables (seleccionable)
            @st.cache_data(show_spinner=False)
            def plot_numeric_histogram(df, selected_col):
                fig, ax = plt.subplots(figsize=(5,3))
                sns.histplot(df[selected_col], bins=50, kde=True, ax=ax, color="#337ab7")
                ax.set_title(f"Histograma de {selected_col}")
                return fig
            with col2:
                st.subheader("Distribución de variables numéricas")
                numeric_cols = get_numeric_cols(df)
                selected_col = st.selectbox("Selecciona una variable para ver su histograma:", numeric_cols)
                st.pyplot(plot_numeric_histogram(df, selected_col))

            col3, col4 = st.columns(2)
            # 3. Heatmap de correlación
            @st.cache_data(show_spinner=False)
            def plot_corr_heatmap(df):
                fig, ax = plt.subplots(figsize=(6,5))
                corr = df[get_numeric_cols(df)].corr()
                sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
                ax.set_title("Correlación entre variables numéricas")
                return fig
            with col3:
                st.subheader("Heatmap de correlación")
                st.pyplot(plot_corr_heatmap(df))

            # 4. Análisis temporal
            @st.cache_data(show_spinner=False)
            def plot_traffic_per_day(df):
                df_copy = df.copy()
                df_copy["Timestamp"] = pd.to_datetime(df_copy["Timestamp"], errors="coerce")
                df_copy["date"] = df_copy["Timestamp"].dt.date
                traffic_per_day = df_copy.groupby("date").size()
                fig, ax = plt.subplots(figsize=(6,3))
                traffic_per_day.plot(kind='line', marker='o', ax=ax)
                ax.set_title("Tráfico total por día")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Cantidad de registros")
                plt.xticks(rotation=45)
                return fig
            if "Timestamp" in df.columns:
                with col4:
                    st.subheader("Tráfico total por día")
                    st.pyplot(plot_traffic_per_day(df))

            # 5 y 6. Gráficas temporales lado a lado
            @st.cache_data(show_spinner=False)
            def plot_hourly_benign_attack(df):
                df_copy = df.copy()
                df_copy["Timestamp"] = pd.to_datetime(df_copy["Timestamp"], errors="coerce")
                df_copy["hour"] = df_copy["Timestamp"].dt.hour
                fig, ax = plt.subplots(figsize=(6,3))
                df_copy.groupby(['hour', 'Label']).size().unstack(fill_value=0).plot(kind='bar', stacked=True, ax=ax, color=['#5cb85c', '#d9534f'])
                ax.set_title("Registros por hora")
                ax.set_xlabel("Hora del día")
                ax.set_ylabel("Cantidad")
                return fig
            @st.cache_data(show_spinner=False)
            def plot_hourly_attack_type(df):
                df_copy = df.copy()
                df_copy["Timestamp"] = pd.to_datetime(df_copy["Timestamp"], errors="coerce")
                df_copy["hour"] = df_copy["Timestamp"].dt.hour
                if "Attack Name" in df_copy.columns:
                    traffic_per_attack = df_copy.groupby(["hour", "Attack Name"]).size().unstack(fill_value=0)
                    fig, ax = plt.subplots(figsize=(6,3))
                    traffic_per_attack.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
                    ax.set_title("Traffic per Hour by Attack Type")
                    ax.set_xlabel("Hour of Day")
                    ax.set_ylabel("Count")
                    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
                    return fig
                return None
            if "Timestamp" in df.columns:
                col_time1, col_time2 = st.columns(2)
                with col_time1:
                    st.subheader("Registros por hora: Benign vs Attack")
                    st.pyplot(plot_hourly_benign_attack(df))
                with col_time2:
                    st.subheader("Tráfico por hora por tipo de ataque")
                    fig_attack = plot_hourly_attack_type(df)
                    if fig_attack:
                        st.pyplot(fig_attack)
        st.markdown("---")
        st.subheader("Conclusiones del Análisis Exploratorio")
        st.markdown("""
    **Principales conclusiones basadas en las visualizaciones:**

    - De la gráfica de distribución de clases, se observa un fuerte desbalance: los ataques (especialmente DoS) dominan el dataset y los registros benignos son minoría.
    - El histograma de variables numéricas permite identificar outliers y rangos característicos para cada tipo de tráfico.
    - El heatmap de correlación evidencia que muchas variables están altamente correlacionadas, lo que justifica la reducción de características redundantes.
    - El análisis temporal muestra que la mayoría de los ataques se concentran entre la 1 y 3 PM, mientras que no existe un patrón claro a nivel de días.
    - Las gráficas de registros por hora y tráfico por tipo de ataque permiten identificar los momentos de mayor actividad maliciosa y los ataques predominantes en cada franja horaria.

    **Para el preprocesamiento, se conservarán solo las variables esenciales y discriminativas:**
    - Comportamiento de paquetes: `Fwd Packet Length Mean`, `Bwd Packet Length Mean`
    - Comportamiento temporal: `Fwd IAT Mean`, `Bwd IAT Mean`
    - Descriptor global de tamaño de paquete: `Packet Length Mean`
    - Indicadores de actividad: `Active Mean`, `Idle Mean`
    - Flags TCP: `SYN`, `ACK`, `RST`, `URG`
    - IP de destino (clave para identificar objetivos repetidos)
    - Puertos (`Src Port`, `Dst Port`)
    - Contadores de flujo (Subflow stats, Total Fwd/Bwd Packet)

    Estas decisiones optimizan el dataset para el modelado, eliminando ruido y redundancia, y asegurando que las variables seleccionadas sean realmente útiles para la detección de ataques en IoT.
        """)

with model_tab:
    st.header("Resultados del Modelo")
    st.markdown("""
    Se presentan las métricas de desempeño y análisis del modelo Random Forest:
    - Accuracy, F1-score, F1-macro
    - Matriz de confusión
    - Importancia de características
    """)
    col_mod1, col_mod2 = st.columns([2, 1])
    with col_mod1:
        st.subheader("Importancia de características")
        # Ajustar ruta y tamaño de la imagen, y manejo de error
        import os
        img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports/figures/rf_feature_importances.png'))
        if os.path.exists(img_path):
            st.image(img_path, caption="Top 20 características importantes", use_container_width=True)
        else:
            st.warning(f"No se encontró la imagen de importancias en: {img_path}")
    with col_mod2:
        st.subheader("Métricas de evaluación")
        metrics_data = {
            "Métrica": ["Accuracy", "Macro F1", "Weighted F1"],
            "Valor": [0.9973, 0.774, 0.9975]
        }
        st.table(pd.DataFrame(metrics_data))

    st.subheader("Reporte de clasificación")
    report_data = [
        ["Benign Traffic", 0.97, 0.98, 0.98, 6521],
        ["DDoS ICMP Flood", 0.49, 0.55, 0.52, 506],
        ["DDoS UDP Flood", 0.53, 0.59, 0.56, 510],
        ["DoS ICMP Flood", 0.49, 0.55, 0.52, 421],
        ["DoS TCP Flood", 1.00, 1.00, 1.00, 421381],
        ["DoS UDP Flood", 0.75, 0.76, 0.75, 618],
        ["MITM ARP Spoofing", 0.39, 0.49, 0.43, 211],
        ["MQTT DDoS Publish Flood", 1.00, 1.00, 1.00, 60968],
        ["MQTT DoS Connect Flood", 1.00, 1.00, 1.00, 47606],
        ["MQTT DoS Publish Flood", 0.86, 0.91, 0.89, 191],
        ["MQTT Malformed", 0.71, 0.87, 0.78, 449],
        ["Recon OS Scan", 1.00, 0.99, 0.99, 17063],
        ["Recon Ping Sweep", 0.16, 0.29, 0.21, 14],
        ["Recon Port Scan", 1.00, 1.00, 1.00, 97104],
        ["Recon Vulnerability Scan", 0.99, 0.99, 0.99, 1664],
    ]
    report_df = pd.DataFrame(report_data, columns=["Clase", "Precisión", "Recall", "F1-score", "Soporte"])
    st.dataframe(report_df, height=400)

    st.markdown("---")
    st.subheader("Conclusiones sobre el modelo y las características")
    st.markdown("""
**Interpretación de las características más relevantes en la detección de ataques IoT:**

- **Flow Duration**: Es la variable más influyente, separa ataques de tipo flooding, reconocimiento y tráfico benigno por la duración de los flujos.
- **Inter-Arrival Times (IAT)**: Fwd IAT Mean y Bwd IAT Mean capturan patrones temporales característicos de cada tipo de ataque y tráfico.
- **Bwd Init Win Bytes**: El tamaño de ventana TCP diferencia entre dispositivos IoT, malware y ataques de flooding.
- **Puertos de origen y destino**: Permiten identificar ataques de escaneo, tráfico MQTT y ataques dirigidos a servicios específicos.
- **TCP Flag Counts (SYN, ACK, RST)**: Detectan anomalías de bajo nivel en el transporte, como SYN Floods y patrones de reconocimiento.
- **Packet Length Mean**: Distingue entre bots, tráfico de reconocimiento, MQTT y tráfico benigno por el tamaño promedio de los paquetes.
- **DstIP_Encoded**: Indica que los ataques suelen dirigirse a dispositivos específicos, pero la codificación evita el sobreajuste.
- **Active Mean & Idle Mean**: Capturan el ritmo de actividad y reposo, diferenciando ataques continuos, ráfagas de reconocimiento y tráfico benigno.

**Interpretación general:**

Las características más importantes reflejan patrones temporales, comportamiento en la capa de transporte, targeting de servicios y estructura de los paquetes. Esto confirma que:

✔ Las variables seleccionadas retienen información relevante y discriminativa.
✔ El preprocesamiento no eliminó discriminadores críticos.
✔ Las decisiones del modelo se basan en patrones reales de red.
✔ El Random Forest aprende la estructura verdadera y no ruido.

El modelo Random Forest logra una alta precisión y F1-score ponderado, detectando casi perfectamente las clases dominantes y mostrando capacidad discriminativa incluso en clases minoritarias. Aunque algunas clases raras tienen menor recall, el comportamiento general indica que los patrones de ataque están bien capturados y que los modelos no lineales son adecuados para este problema.
    """)
    
with predict_tab:
    st.header("Predicción en línea")
    st.markdown("""
    Ingresa un JSON con las características extraídas de un flujo de red IoT. El sistema aplicará el preprocesamiento y mostrará la predicción del modelo entrenado.
    """)
    user_json = st.text_area("Pega aquí el JSON de características:", "{}", height=200)
    if st.button("Predecir"):
        try:
            input_data = json.loads(user_json)
            # Convert JSON to DataFrame
            df_pred = pd.DataFrame([input_data])

            # 1. remove irrelevant columns
            irrelevant_features = [
                "Protocol", "Bwd PSH Flags", "Bwd URG Flags", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
                "PSH Flag Count", "FIN Flag Count", "ECE Flag Count", "CWR Flag Count"
            ]
            df_pred.drop(columns=[col for col in irrelevant_features if col in df_pred.columns], inplace=True)

            # 2. remove multicollinears
            redundant_features = [
                "Bwd Packet Length Max", "Bwd Packet Length Std", "Bwd Segment Size Avg", "Total Length of Bwd Packet", "Bwd IAT Total", "Bwd IAT Max", "Bwd IAT Std",
                "Fwd Packet Length Max", "Fwd Packet Length Std", "Fwd Segment Size Avg", "Total Length of Fwd Packet", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std",
                "Packet Length Max", "Packet Length Std", "Packet Length Variance", "Average Packet Size",
                "Idle Max", "Idle Min", "Idle Std", "Active Max", "Active Min", "Active Std",
                "Fwd Act Data Pkts", "Fwd Header Length", "Bwd Header Length", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
                "Flow Packets/s", "Bwd Packets/s", "Fwd Packets/s", "Flow Bytes/s", "Bwd IAT Min", "Fwd IAT Min"
            ]
            df_pred.drop(columns=[col for col in redundant_features if col in df_pred.columns], inplace=True)

            # 3. remove identifiers and noise columns
            id_columns = ["Flow ID", "Timestamp", "source_file", "Src IP"]
            df_pred.drop(columns=[col for col in id_columns if col in df_pred.columns], inplace=True)

            # 4. standardize numeric features
            numeric_features = df_pred.select_dtypes(include=["int64", "float64"]).columns.tolist()
            numeric_features = [c for c in numeric_features if c not in ["Label", "Attack Name"]]
            scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/scaler.pkl'))
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                df_pred[numeric_features] = scaler.transform(df_pred[numeric_features])
            else:
                scaler = StandardScaler()
                df_pred[numeric_features] = scaler.fit_transform(df_pred[numeric_features])

            # 5. Encode Dst IP safely
            le_ip_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/label_encoder_dst_ip.pkl'))

            if "Dst IP" in df_pred.columns and os.path.exists(le_ip_path):
                with open(le_ip_path, "rb") as f:
                    le_ip = pickle.load(f)

                ip_value = df_pred["Dst IP"].iloc[0]

                # Safe encoding
                if ip_value in le_ip.classes_:
                    df_pred["DstIP_Encoded"] = le_ip.transform([ip_value])[0]
                else:
                    # Assign a safe "unknown" encoded value
                    unknown_value = len(le_ip.classes_)
                    df_pred["DstIP_Encoded"] = unknown_value

                # Remove raw column
                df_pred.drop(columns=["Dst IP"], inplace=True)


            # 6. charge model and predict
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/random_forest_best.pkl'))
            if not os.path.exists(model_path):
                st.error("No se encontró el modelo entrenado. Ejecuta el script de entrenamiento primero.")
            else:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df_pred.columns.tolist()
                X_pred = df_pred.reindex(columns=model_features, fill_value=0)
                y_pred = model.predict(X_pred)
                le_attack_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/label_encoder_attack_name.pkl'))
                if os.path.exists(le_attack_path):
                    with open(le_attack_path, "rb") as f:
                        le_attack = pickle.load(f)
                    attack_name = le_attack.inverse_transform([y_pred[0]])[0]
                    st.success(f"Predicción del modelo: {attack_name}")
                else:
                    st.success(f"Predicción del modelo (índice): {y_pred[0]}")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
