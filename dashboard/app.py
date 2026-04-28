"""
Training Dashboard

Run with:
    streamlit run dashboard/app.py
"""

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from model.network import Network
from model.presets import CNN_SMALL, MLP_BASELINE
from model.registry import INITIALIZERS, LAYERS

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Training Dashboard", layout="wide")
st.title("Training Dashboard")

# ── session state defaults ────────────────────────────────────────────────────

if "layers" not in st.session_state:
    st.session_state.layers: list[dict] = []

# ── helpers ───────────────────────────────────────────────────────────────────


def _layer_label(cfg: dict) -> str:
    t = cfg["type"]
    if t == "dense":
        return f"Dense  {cfg['input_size']} → {cfg['output_size']}"
    if t == "conv2d":
        return f"Conv2D  {cfg['in_channels']} → {cfg['out_channels']}ch  k={cfg['kernel_size']}"
    if t == "maxpool2d":
        return f"MaxPool2D  k={cfg['kernel_size']}  s={cfg['stride']}"
    if t == "flatten":
        return f"Flatten  [{cfg.get('start_dim', 1)} … {cfg.get('end_dim', -1)}]"
    return t.capitalize()


def _network_summary(layers: list[dict]) -> str:
    try:
        net = Network.from_config(layers)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            net.summary(input_shape=(1, 28, 28))
        return buf.getvalue()
    except Exception as e:
        return f"Error: {e}"


# ── network builder ───────────────────────────────────────────────────────────

st.header("Network Builder")

# — presets ——————————————————————————————————————————————————————————————————

st.subheader("Presets")
pcol1, pcol2, _ = st.columns([1, 1, 6])
if pcol1.button("MLP Baseline"):
    st.session_state.layers = list(MLP_BASELINE)
    st.rerun()
if pcol2.button("Small CNN"):
    st.session_state.layers = list(CNN_SMALL)
    st.rerun()

st.divider()

# — add layer —————————————————————————————————————————————————————————————————

st.subheader("Add Layer")

initializers = INITIALIZERS.keys()
layer_type = st.selectbox("Layer type", LAYERS.keys(), key="add_type")
cfg: dict = {"type": layer_type}

if layer_type == "dense":
    c1, c2, c3, c4 = st.columns(4)
    cfg["input_size"]  = int(c1.number_input("input_size",  min_value=1, value=128, step=1, key="d_in"))
    cfg["output_size"] = int(c2.number_input("output_size", min_value=1, value=64,  step=1, key="d_out"))
    cfg["bias"]        = c3.checkbox("bias", value=True, key="d_bias")
    cfg["initializer"] = c4.selectbox("initializer", initializers, key="d_init")

elif layer_type == "conv2d":
    c1, c2, c3 = st.columns(3)
    cfg["in_channels"]  = int(c1.number_input("in_channels",       min_value=1, value=1,  step=1, key="c_inc"))
    cfg["out_channels"] = int(c2.number_input("out_channels",      min_value=1, value=8,  step=1, key="c_outc"))
    cfg["kernel_size"]  = int(c3.number_input("kernel_size (odd)", min_value=1, value=3,  step=2, key="c_k"))
    c4, c5, c6 = st.columns(3)
    cfg["stride"]       = int(c4.number_input("stride",  min_value=1, value=1, step=1, key="c_s"))
    cfg["padding"]      = int(c5.number_input("padding", min_value=0, value=0, step=1, key="c_p"))
    cfg["bias"]         = c6.checkbox("bias", value=True, key="c_bias")
    cfg["initializer"]  = st.selectbox("initializer", initializers, key="c_init")

elif layer_type == "maxpool2d":
    c1, c2, c3 = st.columns(3)
    cfg["kernel_size"] = int(c1.number_input("kernel_size", min_value=1, value=2, step=1, key="mp_k"))
    cfg["stride"]      = int(c2.number_input("stride",      min_value=1, value=2, step=1, key="mp_s"))
    cfg["padding"]     = int(c3.number_input("padding",     min_value=0, value=0, step=1, key="mp_p"))

elif layer_type == "flatten":
    c1, c2 = st.columns(2)
    cfg["start_dim"] = int(c1.number_input("start_dim", value=1,  step=1, key="fl_s"))
    cfg["end_dim"]   = int(c2.number_input("end_dim",   value=-1, step=1, key="fl_e"))

elif layer_type == "softmax":
    cfg["dim"] = int(st.number_input("dim", value=-1, step=1, key="sm_dim"))

# relu, sigmoid: no configurable params

if st.button("Add layer", type="primary"):
    st.session_state.layers.append(dict(cfg))
    st.rerun()

st.divider()

# — layer list ————————————————————————————————————————————————————————————————

st.subheader("Layers")

if not st.session_state.layers:
    st.caption("No layers yet. Add one above or load a preset.")
else:
    n = len(st.session_state.layers)
    for i, layer in enumerate(st.session_state.layers):
        lc, uc, dc, rc = st.columns([5, 1, 1, 1])
        lc.write(f"**{i}** — {_layer_label(layer)}")

        if uc.button("↑", key=f"up_{i}", disabled=(i == 0)):
            st.session_state.layers[i - 1], st.session_state.layers[i] = (
                st.session_state.layers[i],
                st.session_state.layers[i - 1],
            )
            st.rerun()

        if dc.button("↓", key=f"dn_{i}", disabled=(i == n - 1)):
            st.session_state.layers[i + 1], st.session_state.layers[i] = (
                st.session_state.layers[i],
                st.session_state.layers[i + 1],
            )
            st.rerun()

        if rc.button("✕", key=f"rm_{i}"):
            st.session_state.layers.pop(i)
            st.rerun()

st.divider()

# — network summary ————————————————————————————————————————————————————————————

st.subheader("Summary")

if not st.session_state.layers:
    st.caption("Add layers to see the summary.")
else:
    summary = _network_summary(st.session_state.layers)
    if summary.startswith("Error:"):
        st.error(summary)
    else:
        st.code(summary)
