import streamlit as st
from datasafe.storage.sqlite_store import query_recent

st.set_page_config(page_title="DataSafe — Media", layout="wide")
st.title("Media — Active Threats & Actions")

col1,col2,col3 = st.columns(3)
severity = col1.selectbox("Severity", ["All","Critical","High","Medium","Low"], index=0)
min_risk = col2.slider("Min digital risk", 0, 100, 30, 10)
limit = col3.slider("Max items", 50, 1000, 200, 50)

sev=None if severity=="All" else severity
rows=query_recent(limit=limit, severity=sev, min_risk=min_risk)

st.caption(f"{len(rows)} items")

for r in rows:
    with st.container(border=True):
        st.subheader(r.get("title") or "(untitled)")
        st.write(r.get("summary") or "")
        st.markdown(f"**Source:** {r['source']} | **Severity:** {r['severity']} | **Risk(d/p):** {r['risk']['digital']}/{r['risk']['physical']}")
        if r.get("url"):
            st.markdown(f"[Source link]({r['url']})")
        st.markdown("**Actions (contextual):**")
        for a in (r.get("actions") or []):
            st.markdown(f"- {a}")
        with st.expander("Indicators & Labels"):
            st.write("Threat labels:", ", ".join(r.get("threats") or []))
            st.write("Sectors:", ", ".join(r.get("sectors") or []))
            st.json(r.get("iocs") or {})
import streamlit as st
from typing import Optional
from datasafe.storage.sqlite_store import query_recent

st.title("Media Risks")

severity = st.selectbox("Severity", ["All","Critical","High","Medium","Low"], index=0)
min_risk = st.slider("Min Risk", 0, 100, 50)

limit = st.number_input("Limit", 10, 500, 200)

sector = "Media"

items = query_recent(limit=int(limit), sector=sector, severity=None if severity=="All" else severity, min_risk=min_risk)

st.subheader("Top Active Risks")
for it in items[:10]:
    risk = it.get("risk", {})
    st.metric(label=it.get("title", ""), value=f"D:{risk.get('digital',0)} P:{risk.get('physical',0)}", delta=it.get("severity"))
    st.caption(it.get("summary", ""))
    if it.get("url"):
        st.write(it.get("url"))

st.subheader("Threat Table")
st.dataframe(items)


