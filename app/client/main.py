import datetime
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st


SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8500")
REQUEST_TIMEOUT = 10

RADIO_LICHESS_LINK_TEXT = "Lichess tournament link or ID"
RADIO_TSV_FILE_TEXT = "TSV dataset file"


def try_request(method: str, url: str, **kwargs) -> requests.Response:
    """Tries to send a request to the server. If an error occurs, displays an error message and stops the script."""
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
    except requests.Timeout:
        st.error("Server timed out")
        st.stop()
    except requests.HTTPError as e:
        try:
            st.error(response.json()["detail"])
        except requests.exceptions.JSONDecodeError:
            st.error(str(e))
        st.stop()
    except requests.RequestException:
        st.error("Could not connect to server")
        st.stop()
    return response


st.set_page_config(
    page_title="Lichess Arena Attendance Prediction",
    page_icon=":chess_pawn:"
)
st.header("Lichess Arena Attendance Prediction:chess_pawn:")

if "model_names" not in st.session_state:
    st.session_state.model_names = try_request("GET", f"{SERVER_URL}/list_models").json()
model_name = st.segmented_control(
    "Select model",
    options=st.session_state.model_names,
    default=st.session_state.model_names[0]
)

source = st.radio(
    "Predict from:",
    options=[RADIO_LICHESS_LINK_TEXT, RADIO_TSV_FILE_TEXT],
    captions=[
        "Predict how many players will participate in a given tournament. The prediction can be explained with LIME.",
        "Make predictions for an entire dataset. If true attendance figures are provided, compute regression metrics.",
    ]
)

if source == RADIO_LICHESS_LINK_TEXT:
    link_or_id = st.text_input("Lichess tournament link or ID")

    st.write("")
    _, center, _ = st.columns(3)
    predict_pressed = center.button(
        "Predict", type="primary", disabled=not (model_name and link_or_id), use_container_width=True
    )
    st.write("")

    if predict_pressed:
        st.session_state.pop("predict_link_response", None)
        st.session_state.pop("altair_chart", None)
        st.session_state.predict_link_response = try_request(
            "POST", f"{SERVER_URL}/predict_link/{model_name}", params={"tournament_link_or_id": link_or_id}
        ).json()

    if st.session_state.get("predict_link_response"):
        response = st.session_state.predict_link_response

        starts_at = datetime.datetime.fromisoformat(response["starts_at"]).astimezone().strftime("%b %e %Y %H:%M")
        st.subheader(f"{response["name"]} :gray[{starts_at}]")

        col1, col2 = st.columns(2)
        col1.metric("Predicted attendance", f"{response["n_players_pred"]} players")
        if response["n_players_true"] is not None:
            col2.metric("Actual attendance", f"{response["n_players_true"]} players")

        _, center, _ = st.columns(3)
        if "altair_chart" in st.session_state:
            st.write("Most important features for this prediction according to LIME:")
            st.altair_chart(st.session_state.altair_chart, use_container_width=True)
        elif center.button("Explain this prediction", use_container_width=True):
            data = try_request(
                "POST", f"{SERVER_URL}/explain/{model_name}", params={"tournament_link_or_id": link_or_id}
            ).json()
            df = pd.DataFrame(data)

            st.session_state.altair_chart = alt.Chart(df).encode(
                y=alt.X("feature", sort=None),
                x=alt.Y("weight"),
                color=alt.condition(alt.datum.weight > 0, alt.ColorValue("green"), alt.ColorValue("red"))
            ).mark_bar().configure_axis(labelLimit=300)
            st.rerun()

else:
    tsv_file = st.file_uploader("Upload dataset file", type="tsv")
    _, center, _ = st.columns(3)
    predict_pressed = center.button(
        "Predict", type="primary", disabled=not (model_name and tsv_file), use_container_width=True
    )

    if predict_pressed:
        st.session_state.pop("predict_tsv_response", None)
        st.session_state.pop("tsv_file_name", None)
        st.session_state.predict_tsv_response = try_request(
            "POST", f"{SERVER_URL}/predict_tsv/{model_name}", files={"tsv_file": tsv_file}
        ).json()
        st.session_state.tsv_file_name = tsv_file.name

    if st.session_state.get("predict_tsv_response"):
        response = st.session_state.predict_tsv_response

        st.subheader(f"{st.session_state.tsv_file_name} :gray[{len(response["n_players_true"])} samples]")

        if response["metrics"] is not None:
            cols = st.columns(4)
            cols[0].metric("R²", f"{response["metrics"]["r2"]:.3f}")
            cols[1].metric("MAPE", f"{response["metrics"]["mape"]:.1%}")
            cols[2].metric("MAE", f"{response["metrics"]["mae"]:.1f}")
            cols[3].metric("MSE", f"{response["metrics"]["mse"]:.1f}")

        preds_str = "\n".join(str(y) for y in response["n_players_pred"])
        tsv_data = f"n_players_pred\n{preds_str}\n"

        _, center, _ = st.columns(3)
        center.download_button(
            "Download predictions (TSV)", data=tsv_data, file_name="predictions.tsv", use_container_width=True
        )
