from typing import List, Tuple
import streamlit as st
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, RocCurveDisplay, DetCurveDisplay
import httpx
import asyncio
import nest_asyncio
import pandas as pd
from pathlib import Path
nest_asyncio.apply()

BASE_URL = "http://fastapi_backend:8000/"  # Root backend endpoint
EXTERNAL_URL = "http://localhost:8000/"  # Connecting from outside
NEW_YEAR_EDITION = True  # Snow or Ballons


st.set_page_config(
    page_title="Pneumonia recognition",
    page_icon=":material/pulmonology:",
    # layout="wide",
)


# General functions section
async def make_get_request(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response


async def get_list_options(what: str = "models") -> List:
    """
    Get list with names from backend

    Parameters:
    what (str): "models" or "datasets"
    """
    if what == "models":
        url = BASE_URL + "list_models"
    elif what == "datasets":
        url = BASE_URL + "list_datasets"

    response = asyncio.run(make_get_request(url))
    if response.status_code != 200:
        exp = 'Bad response from backend'
        st.error(exp)
        return None
    response = response.json()
    return response[what]


def get_model_by_name():
    """Get from backend model"""
    pass


def success_effect():
    if NEW_YEAR_EDITION:
        st.snow()
    else:
        st.balloons()


def calc_gini():
    roc_auc_metric = 0.9
    gini = 2 * roc_auc_metric - 1
    return gini


# Plot functions section
def conf_mat_plot(model, X_test, y_test):
    """Confusion matrix (static)"""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    LABELS = ["Healthy", "Not Healthy"]
    ax[0].set_title("По кол-ву объектов")
    ax[1].set_title("Нормировка по распространённостям класса")
    disp_1 = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=LABELS,
        cmap="crest",
        ax=ax[0]
        )
    disp_1.im_.set_clim(0)
    disp_2 = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=LABELS,
        cmap="crest",
        normalize="true",
        ax=ax[1]
        )
    disp_2.im_.set_clim(0)


def roc_det_plot(classifiers: List[Tuple]):
    """interactive"""
    pass


def start_page():
    """Start Page"""
    st.title("Диагностика пневмонии на флюорограмме")
    st.header("Годовой проект команды :red[25]")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.page_link(predict_page_object, label='Инференс', icon=":material/smart_toy:")
        st.write("Предобученная модель ищет признаки пневмонии на флюорограмме")

    with col2:
        st.page_link(eda_page_object, label='Аналитика', icon=":material/search_insights:")
        st.write("Краткий обзор датасета, на котором модель училась")

    with col3:
        st.page_link(upload_page_object, label='Эксперименты', icon=":material/biotech:")
        st.write("Здесь можно загружать датасеты, обучать на них новые модели и сравнивать результаты")


def predict_page():
    """Get Predictions Page"""
    st.title("Получение предсказаний на одной картинке")

    multi = """#### Как использовать
    1. Заполнить поля
    2. Нажать на кнопочку 'Отправить'
    3. Увидеть результат
    """
    st.markdown(multi)
    st.divider()
    st.markdown("#### Запросить предсказание")
    st.caption("Выберите модель:")
    model_options = asyncio.run(get_list_options("models"))
    model_option = st.selectbox(
        "Выберите модель, которая будет предсказывать", model_options, label_visibility="collapsed")

    st.caption("Загрузите изображение:")
    file = st.file_uploader(
        label="Загрузить изображение",
        type=["jpeg", "png"],
        help="Загружать одно изображение в .png или .jpeg формате",
        label_visibility="collapsed",
    )

    if st.button("Отправить", type="primary"):
        if file is None:
            exp = ValueError("Загрузите изображение")
            st.exception(exp)
        else:
            payload = {
                "file": file,
                "model_name": model_option
            }
            url = BASE_URL + "predict"
            response = httpx.post(url, files=payload)
            if response.status_code == 200:
                st.divider()
                st.markdown("#### Результат")
                success_effect()

                col1, col2 = st.columns([2, 1])
                with col1:
                    response = response.json()
                    verdict = response["result"][0]
                    st.success(
                        f"{'Обнаружены признаки пневмонии' if verdict else 'Признаков пневмонии не обнаружено'}"
                        )

                    st.caption("Имя модели:")
                    st.code(response["model_name"], wrap_lines=True)
                    st.caption("Все параметры модели:")
                    st.code(response["parameters"], wrap_lines=True)

                with col2:
                    st.image(file, caption="Загруженное изображение", use_container_width=True)
            else:
                exp = ValueError('Response: {response.status_code}')
                st.exception(exp)
                "Код ответа {response.status_code}"

    st.write(":grey[Если у вас нет подходящего изображения, можно его скачать:]")
    # demo_picture_url = f"{BASE_URL}demo_picture"
    demo_picture_url = f"{EXTERNAL_URL}demo_picture"
    st.link_button("Скачать случайную демо-картинку", demo_picture_url)


def run_ex_page():
    """Run Experiments Page"""
    st.title("Модели и Датасеты")
    st.divider()

    multi = """#### Загрузка датасетов
    Архив должен содержать:
    1. Изображения в допустимых форматах: png, jpg, jpeg
    2. labels.csv
    """
    st.markdown(multi)

    st.caption("Введите имя вашего датасета")
    dataset_name = st.text_input(
        label="dataset",
        value="My Dataset",
        max_chars=50,
        label_visibility="collapsed"
    )
    dataset_name = dataset_name.strip()

    st.caption("Загрузите .zip file")
    dataset_file = st.file_uploader(
        label="Загрузить датасет",
        type=["zip", "x-zip-compressed"],
        label_visibility="collapsed",
    )

    if st.button("Отправить на загрузку", type="primary"):
        curr_ds = asyncio.run(get_list_options("datasets"))
        if dataset_file is None:
            st.error("Загрузите zip архив")
        if len(dataset_name) == 0:
            st.error("Имя датасета не должно быть пустым")
        if dataset_name in curr_ds:
            st.error("Это имя датасета уже занято")
            st.info(f"Список уже занятых имён: {curr_ds}")
        else:
            url = BASE_URL + "add_dataset"
            files = {
                "file": dataset_file
            }
            # response = httpx.post(url, data=data, files=files, timeout=30.0)
            response = httpx.post(url, files=files, params={"dataset_name": dataset_name}, timeout=30.0)
            if response.status_code == 200:
                success_effect()
                text = response.json()["message"]
                st.success(text)
            else:
                response = response.json()
                exp = ValueError(f"{response}")
                st.exception(exp)

    st.write(":grey[Если у вас нет подходящего датасета, можно скачать готовый демо-датасет:]")
    # demo_dataset_url = f"{BASE_URL}demo_dataset"
    demo_dataset_url = f"{EXTERNAL_URL}demo_dataset"
    st.link_button("Скачать демо-датасет", demo_dataset_url)

    st.divider()
    st.markdown("#### Список доступных датасетов")

    dataset_options = asyncio.run(get_list_options("datasets"))

    st.write(pd.DataFrame(dataset_options, columns=["Загруженные датасеты"]))

    st.divider()

    with st.form('add_model'):
        st.markdown("#### Выбор опций обучения модели")

        st.caption("Введите имя вашей модели")
        model_name = st.text_input(
            label="model_name",
            value="My Model",
            max_chars=50,
            label_visibility="collapsed"
        )

        st.caption("Число деревьев в ансамбле")
        n_estimators = st.number_input(
            label="n_estimators",
            min_value=1,
            step=1,
            value=100,
            label_visibility="collapsed"
        )

        st.caption("Максимальная глубина дерева в ансамбле")
        max_depth = st.number_input(
            label="max_depth",
            # min_value=1,
            step=1,
            value=100,
            label_visibility="collapsed"
        )

        st.caption("Random state")
        rs = st.number_input(
            label="random_state",
            # min_value=1,
            step=1,
            value=74,
            label_visibility="collapsed"
        )

        dataset_options = asyncio.run(get_list_options("datasets"))
        st.caption("Выберите один датасет из загруженных")
        dataset_option = st.selectbox(
            "Выберите датасет", dataset_options, label_visibility="collapsed")

        st.divider()
        submitted = st.form_submit_button("Обучить модель", type="primary")
        if submitted:
            curr_md = asyncio.run(get_list_options("models"))
            if len(model_name) == 0:
                exp = ValueError("Имя модели не должно быть пустым")
                st.exception(exp)
            if model_name in curr_md:
                exp = ValueError("Это имя модели уже занято")
                st.exception(exp)
                st.write(f"Список уже занятых имён: {curr_md}")
            if max_depth == 0 or max_depth < -1:
                error_text = "Недопустимое значение. Можно положительные целые числа и '-1'"
                st.error(error_text)
            else:
                url = BASE_URL + "add_model"
                payload = {
                    'model_name': model_name,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': rs,
                    'dataset_name': dataset_option
                    }
                response = httpx.post(url, params=payload, timeout=30.0)
                if response.status_code == 200:
                    success_effect()
                    response = response.json()["message"]
                    st.success(response)
                else:
                    response = response.json()
                    exp = ValueError(f"{response}")
                    st.exception(exp)

        # res = True
        # if res:
        #     st.success("Обучение завершено")
        #     success_effect()
        # else:
        #     exp = ValueError('This is an exception of type ValueError')
        #     st.exception(exp)
            st.divider()
    st.markdown("#### Список доступных моделей")

    dataset_options = asyncio.run(get_list_options("models"))

    st.write(pd.DataFrame(dataset_options, columns=["Загруженные модели"]))


def compare_page():
    """Comparative Models Analysis Page"""
    def get_model_scores():
        """Fetch model scores for all stored models."""
        response = requests.get(f"{BASE_URL}/list_models")
        response.raise_for_status()
        model_names = response.json().get("models", [])

        metrics = {}
        for model_name in model_names:
            scores_response = requests.get(f"{BASE_URL}/model_scores?model_name={model_name}")
            scores_response.raise_for_status()
            metrics[model_name] = scores_response.json()

        return metrics

    st.title("Сравнение моделей")

    try:
        model_metrics = get_model_scores()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch model scores: {e}")

    metrics_df = pd.DataFrame(
        {
            "Model": list(model_metrics.keys()),
            "Accuracy": [m["accuracy"] for m in model_metrics.values()],
            "F-beta Score": [m["f2"] for m in model_metrics.values()],
            "ROC-AUC": [m["roc_auc"] for m in model_metrics.values()],
        }
    )

    # Plot Accuracy, F-beta Score, and ROC-AUC as bar plots
    st.subheader("Основные метрики")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    metrics_df.set_index("Model")[["Accuracy", "F-beta Score", "ROC-AUC"]].plot(kind="bar", ax=ax)
    ax.set_title("Сравнение значений основных метрик моделей")
    ax.set_ylabel("Значение")
    ax.set_xlabel("Модель")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Plot combined ROC curves
    st.subheader("ROC-кривые")
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_name, metrics in model_metrics.items():
        RocCurveDisplay(
            fpr=metrics["fpr"],
            tpr=metrics["tpr"],
            roc_auc=metrics["roc_auc"],
            estimator_name=model_name
            ).plot(ax=ax)
    ax.set_title("Сравнение ROC-кривых всех моделей")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.write("### Общая таблица")
    st.dataframe(metrics_df)


def eda_page():
    """Run EDA Page"""
    eda_markdown = Path('EDA.md').read_text(encoding='utf-8')
    st.markdown(eda_markdown, unsafe_allow_html=True)


def deep_page():
    """Run Experiments Page"""
    st.title("Подробные параметры модели")
    try:
        model_names = asyncio.run(get_list_options("models"))
        selected_model = st.selectbox("Выберите модель:", model_names)
        if selected_model:
            st.subheader(f"Параметры {selected_model}:")
            try:
                url = BASE_URL + "model_params"
                params = {
                    'model_name': selected_model,
                }
                model_params_response = requests.get(url, params=params)
                model_params = model_params_response.json()
                st.json(model_params)  # Display parameters as JSON
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch parameters for {selected_model}: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


start_page_object = st.Page(start_page, title="Начало")
predict_page_object = st.Page(predict_page, title="Получение предсказаний")
eda_page_object = st.Page(eda_page, title="Разведочный анализ данных")
upload_page_object = st.Page(run_ex_page, title="Добавление моделей и датасетов")
compare_page_object = st.Page(compare_page, title="Сравнение моделей")
param_page_object = st.Page(deep_page, title="Параметры моделей")

# Pages and Navigation Settings
pg = st.navigation({
    "": [
        start_page_object,
    ],
    "Инференс": [
        predict_page_object,
    ],
    "Аналитика": [
        eda_page_object
    ],
    "Эксперименты": [
        upload_page_object,
        compare_page_object,
        param_page_object,
    ],
})
pg.run()
