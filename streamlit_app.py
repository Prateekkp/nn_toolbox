from pathlib import Path
import streamlit as st

try:
    from src.ui.perceptron_ui import perceptron_page
    from src.ui.forward_propagation import forward_propagation_page
    from src.ui.backward_propagation import backward_propagation_page
    from src.ui.mlp import mlp_page
    from src.open_cv.open_cv_detection import opencv_detection_page

    from src.assets.documnets.perceptron import perceptron_docs_page
    from src.assets.documnets.forward_propagation import forward_propagation_docs_page
    from src.assets.documnets.back_propagation import back_propagation_docs_page
    from src.assets.documnets.mnp import mnp_docs_page

except ModuleNotFoundError as e:
    st.error("🚨 Module loading failed.")
    st.code(str(e))
    st.info("Make sure all folders contain __init__.py and the package is installed correctly.")
    st.stop()

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
        "Backward Propagation",
        "Multi-Layer Perceptron (MNP)",
        "OpenCV"
    ]
)

st.sidebar.info(
    "To choose a module first set below to 'None'"
)


st.sidebar.markdown("---")

st.sidebar.subheader("Documentation")
doc_page = st.sidebar.radio(
    "Select Documentation",
    ["None", "Perceptron", 
     "Forward Propagation", 
     "Backward Propagation", 
     "Multi-Layer Perceptron (MNP)"],
    key="doc_nav"
)

# Spacer to push content down
# st.sidebar.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)

# if st.sidebar.button("Reset Session"):
#     st.session_state.clear()
#     if hasattr(st, "rerun"):
#         st.rerun()
#     else:
#         st.experimental_rerun()

# ---------------------------
# Main Content Routing
# ---------------------------
if doc_page != "None":
    page = f"Docs - {doc_page}"

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

    PROJECT_ROOT = Path(__file__).resolve().parent
    image_path = PROJECT_ROOT / "src" / "assets" / "nn_image.jpg"

    if not image_path.exists():
        st.warning("⚠️ Image not found. Skipping image rendering.")
    else:
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
    forward_propagation_page()

elif page == "Backward Propagation":
    backward_propagation_page()

elif page == "Multi-Layer Perceptron (MNP)":
    mlp_page()

elif page == "OpenCV":
    opencv_detection_page()

elif page == "Docs - Perceptron":
    perceptron_docs_page()

elif page == "Docs - Forward Propagation":
    forward_propagation_docs_page()

elif page == "Docs - Backward Propagation":
    back_propagation_docs_page()

elif page == "Docs - Multi-Layer Perceptron (MNP)":
    mnp_docs_page()

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

def main():
    import sys
    from pathlib import Path
    from streamlit.web import cli as stcli

    app_path = Path(__file__).resolve()

    # Reset argv EXACTLY like Streamlit expects
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.runOnSave=false",
    ]

    sys.exit(stcli.main())