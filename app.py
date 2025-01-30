import streamlit as st
import numpy as np
from prediction import predict_focus, load_tflite_model


def main():
    st.title("Smart Study Assistant with TFLite")
    st.subheader("Predict if you should continue studying or take a break")

    st.markdown("### Generate Random Input Data")
    if 'heart_rate' not in st.session_state:
        st.session_state.heart_rate = None
        st.session_state.breathing_rate = None
        st.session_state.stress_level = None

    if st.button("Generate Random Values"):
        st.session_state.heart_rate = np.random.randint(50, 110)
        st.session_state.breathing_rate = np.random.randint(10, 25)
        st.session_state.stress_level = np.random.randint(1, 11)
        st.success(
            f"Generated Values: Heart Rate = {st.session_state.heart_rate}, Breathing Rate = {st.session_state.breathing_rate}, Stress Level = {st.session_state.stress_level}")

    st.markdown("### Refine Input Data (Optional)")
    heart_rate = st.number_input("Heart Rate (bpm)",
                                 value=st.session_state.heart_rate if st.session_state.heart_rate else 70, step=1)
    breathing_rate = st.number_input("Breathing Rate (breaths/min)",
                                     value=st.session_state.breathing_rate if st.session_state.breathing_rate else 15,
                                     step=1)
    stress_level = st.number_input("Stress Level",
                                   value=st.session_state.stress_level if st.session_state.stress_level else 3, step=1)

    st.session_state.heart_rate = heart_rate
    st.session_state.breathing_rate = breathing_rate
    st.session_state.stress_level = stress_level

    if st.button("Predict Focus Level"):
        if heart_rate is None or breathing_rate is None or stress_level is None:
            st.error("Please generate or input the data to make a prediction.")
        else:
            interpreter = load_tflite_model()
            st.write(f"Raw Input Data: Heart Rate = {heart_rate}, Breathing Rate = {breathing_rate}, Stress Level = {stress_level}")
            input_data = np.array([[heart_rate, breathing_rate, stress_level]])

            focus_level_class = predict_focus(interpreter, input_data)
            if focus_level_class == 2:
                st.success("🟢 You should continue studying!")
            elif focus_level_class == 1:
                st.warning("🟠 Study for some time and then take a break.")
            else:
                st.warning("🔴 It's time to take a break.")


if __name__ == "__main__":
    main()
