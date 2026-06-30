import streamlit as st
import torch
import joblib
import numpy as np

st.title("De Novo Molecular Generator")
st.caption("Bigram language model trained on 3,000 SMILES strings · IIT Madras")

@st.cache_resource
def load_model():
    data = joblib.load("bigram_model.pkl")
    N    = data["N"].float()
    P    = (N + 1) / (N + 1).sum(dim=1, keepdim=True)  # smoothed probabilities
    return P, data["stoi"], data["itos"]

P, stoi, itos = load_model()

st.sidebar.header("Generation Settings")
n_molecules = st.sidebar.slider("Number of molecules", 1, 50, 10)
max_len     = st.sidebar.slider("Max SMILES length",  20, 150, 80)
seed        = st.sidebar.number_input("Random seed", value=42, step=1)

def generate_smiles(P, itos, stoi, n, max_len, seed):
    torch.manual_seed(seed)
    results = []
    for _ in range(n):
        out = []
        ix  = 0   # start token
        for _ in range(max_len):
            p  = P[ix]
            ix = torch.multinomial(p, 1, replacement=True).item()
            if ix == 0:   # end token
                break
            out.append(itos[ix])
        results.append("".join(out))
    return results

if st.button("Generate Molecules", type="primary"):
    with st.spinner("Generating..."):
        molecules = generate_smiles(P, itos, stoi, n_molecules, max_len, int(seed))

    st.subheader(f"Generated {len(molecules)} SMILES")

    import pandas as pd
    df = pd.DataFrame({"#": range(1, len(molecules)+1), "SMILES": molecules})
    st.dataframe(df, use_container_width=True, hide_index=True)


    # Show length distribution as a simple stat
    lengths = [len(m) for m in molecules]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg length",  f"{np.mean(lengths):.1f} chars")
    col2.metric("Shortest",    f"{min(lengths)} chars")
    col3.metric("Longest",     f"{max(lengths)} chars")

    st.bar_chart(
        pd.DataFrame({"SMILES length": lengths}),
        y="SMILES length"
    )