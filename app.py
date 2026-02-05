from pathlib import Path

import streamlit as st
from src.ui.perceptron_ui import perceptron_page

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Neural Network Toolbox",
    page_icon="🧑‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Sidebar: Navigation
# ---------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Perceptron",
        "Forward Propagation",
        "Backward Propagation"
    ]
)

st.sidebar.markdown("---")

# Spacer to push content down
st.sidebar.markdown("<div style='height: 45vh;'></div>", unsafe_allow_html=True)

if st.sidebar.button("Reset Session"):
    st.session_state.clear()
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ---------------------------
# Main Content Routing
# ---------------------------
if page == "Home":

    # ---------------------------
    # Header Section (HOME ONLY)
    # ---------------------------
    st.markdown(
        """
        <h1 style="text-align:center;">Neural Network Learning Toolbox</h1>
        <p style="text-align:center; font-size:16px;">
        Build, tune, and understand neural networks from scratch — one concept at a time.
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    image_path = Path(__file__).parent / "src" / "assets" / "nn_image.jpg"
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        st.image(str(image_path), width=1200)

    st.subheader("About This Toolbox")

    st.markdown("""
    This application is an **educational neural network simulator** designed to help
    students understand how neural networks work internally.

    ### What you can do here:
    - Learn neural networks by **building them from scratch**
    - Manually tune learning parameters
    - Upload CSV datasets and experiment
    - Visualize training and learning behavior

    ### How to use:
    1. Select a module from the sidebar  
    2. Upload a dataset (if required)  
    3. Configure parameters  
    4. Train and observe results  
    """)

    st.info("Use the sidebar to navigate through different neural network concepts.")

elif page == "Perceptron":

    # st.subheader("Perceptron Learning Module")
    perceptron_page()

elif page == "Forward Propagation":

    st.subheader("Forward Propagation")

    st.markdown("""
    Forward propagation is the process by which input data flows through
    the network to produce an output.
    """)

    st.warning("Forward propagation logic and visual flow will appear here.")

elif page == "Backward Propagation":

    st.subheader("Backward Propagation")

    st.markdown("""
    Backpropagation is the learning mechanism of neural networks.
    It updates weights using gradients calculated from the loss function.
    """)

    st.warning("Backward propagation and weight update visualization will appear here.")

# ---------------------------
# Footer (Global, Minimal)
# ---------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px;">
    Neural Network Learning Toolbox | Built for educational purposes
    </p>
    """,
    unsafe_allow_html=True
)
