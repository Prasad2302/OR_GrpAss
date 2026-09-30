import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import models as m

st.set_page_config(page_title="HAL Defence Manufacturing – OR Dashboard", page_icon="✈️", layout="wide")


@st.cache_data
def tsp():
    return m.solve_tsp()


st.title("✈️ HAL Defence Manufacturing – Operations Research Dashboard")
st.caption("Seven optimisation models solved live in Python. Inputs are academic planning assumptions, "
           "not HAL internal data.")

tabs = st.tabs(["Overview", "1 · LP", "2 · Goal Programming", "3 · Integer", "4 · Transportation",
                "5 · Assignment", "6 · Nonlinear", "7 · TSP"])

# ---------------------------------------------------------------- Overview
lp, ip, tr, asg, nl, ts = m.solve_lp(), m.solve_ip(), m.solve_transport(), m.solve_assignment(), m.solve_nlp(), tsp()
gp = m.solve_gp([g[2] for g in m.GP_GOALS])
with tabs[0]:
    c = st.columns(6)
    c[0].metric("LP contribution", f"₹{lp['obj']:,.1f} L")
    c[1].metric("Sourcing cost", f"₹{ip['cost']:,.0f} L")
    c[2].metric("Transport cost", f"₹{tr['cost']:,.2f} L")
    c[3].metric("Assignment time", f"{asg['total']} h")
    c[4].metric("NLP min cost", f"₹{nl['cost']:,.0f} Cr")
    c[5].metric("TSP route", f"{ts['total']:,} km")
    st.dataframe(pd.DataFrame({
        "Technique": ["Linear Programming", "Goal Programming", "Integer Programming", "Transportation",
                      "Assignment", "Nonlinear Programming", "Travelling Salesman"],
        "Application": ["Aircraft production mix", "Multi-goal production plan", "Critical-component supplier selection",
                        "Supplier → HAL facility allocation", "Engineer → MRO job allocation",
                        "Capacity utilisation vs operating cost", "MRO / inspection service route"],
        "Objective": ["Max contribution", "Min weighted deviation", "Min sourcing cost", "Min transport cost",
                      "Min processing time", "Min operating cost", "Min distance"]}), hide_index=True)

# ---------------------------------------------------------------- 1 LP
with tabs[1]:
    st.subheader("Aircraft production mix")
    with st.expander("Change annual resource availability (%)"):
        cols = st.columns(5)
        pct = [cols[i].slider(r, 50, 150, 100, 5, key=f"lp{i}") for i, r in enumerate(m.RESOURCES)]
    r = m.solve_lp(np.array(m.LP_AVAIL) * np.array(pct) / 100)
    k = st.columns(3)
    k[0].metric("Total contribution", f"₹{r['obj']:,.1f} lakh")
    k[1].metric("Aircraft produced", f"{r['x'].sum():.2f}")
    k[2].metric("Binding resources", int((r["used"] >= r["avail"] - 1e-6).sum()))
    a, b = st.columns(2)
    a.plotly_chart(px.bar(pd.DataFrame({"Product": m.PRODUCTS * 2, "Units": list(r["x"]) + m.LP_DEMAND,
                                        "Series": ["Optimal"] * 4 + ["Max demand"] * 4}),
                          x="Product", y="Units", color="Series", barmode="group", title="Optimal units vs demand"))
    util = 100 * r["used"] / r["avail"]
    fig = px.bar(x=m.RESOURCES, y=util, labels={"x": "", "y": "Utilisation %"}, title="Resource utilisation (%)")
    fig.add_hline(y=100, line_dash="dash", line_color="red")
    b.plotly_chart(fig)
    st.dataframe(pd.DataFrame({"Resource": m.RESOURCES, "Used": r["used"].round(1), "Available": r["avail"],
                               "Slack": (r["avail"] - r["used"]).round(1),
                               "Shadow price (₹ lakh per extra unit)": r["shadow"].round(4)}), hide_index=True)

# ---------------------------------------------------------------- 2 GP
with tabs[2]:
    st.subheader("Multi-goal production plan")
    st.caption("Goal coefficients are assumptions linked to the LP data. Priority 1 carries the highest weight.")
    cols = st.columns(5)
    tg = [cols[i].number_input(f"{g[0]} {g[1]}", value=float(g[2]), key=f"gp{i}") for i, g in enumerate(m.GP_GOALS)]
    g = m.solve_gp(tg)
    k = st.columns(3)
    k[0].metric("Total weighted deviation", f"{g['total']:,.2f}")
    k[1].metric("Goals met", f"{int((g['dev'] < 1e-6).sum())} / 5")
    k[2].metric("Aircraft produced", f"{g['x'].sum():.2f}")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(x=m.PRODUCTS, y=g["x"], labels={"x": "", "y": "Units"}, title="Production plan"))
    pctg = 100 * g["achieved"] / np.array(tg)
    fig = px.bar(x=[x[0] for x in m.GP_GOALS], y=pctg, labels={"x": "", "y": "% of target"}, title="Achieved as % of target")
    fig.add_hline(y=100, line_dash="dash")
    b.plotly_chart(fig)
    st.dataframe(pd.DataFrame({"Goal": [x[0] for x in m.GP_GOALS], "Direction": [x[1] for x in m.GP_GOALS],
                               "Target": tg, "Achieved": g["achieved"].round(2), "Deviation": g["dev"].round(2),
                               "Priority": range(1, 6)}), hide_index=True)

# ---------------------------------------------------------------- 3 IP
with tabs[3]:
    st.subheader("Critical-component supplier selection")
    c1, c2 = st.columns(2)
    ms = c1.slider("Maximum number of suppliers", 3, 8, 7)
    dom = c2.slider("Minimum domestic sourcing (%)", 0, 100, 80) / 100
    r = m.solve_ip(ms, dom)
    if r is None:
        st.error(f"INFEASIBLE with at most {ms} suppliers. Every category needs coverage and two categories "
                 "need two suppliers each, so at least 7 suppliers are required (original limit of 5 fails).")
    else:
        k = st.columns(3)
        k[0].metric("Total cost", f"₹{r['cost']:,.0f} lakh")
        k[1].metric("Suppliers selected", int(r["y"].sum()))
        k[2].metric("Fixed contract cost", f"₹{r['fix'].sum():,.0f} lakh")
        df = pd.DataFrame({"Supplier": m.SUPPLIERS, "Category": m.IP_CAT, "Units": r["q"], "Capacity": m.IP_CAP,
                           "Contract": np.where(r["y"] > 0, "Yes", "No"), "Cost (₹ lakh)": r["var"] + r["fix"]})
        st.plotly_chart(px.bar(df, x="Supplier", y="Units", color="Category", title="Units sourced per supplier"))
        st.dataframe(df, hide_index=True)

# ---------------------------------------------------------------- 4 Transportation
with tabs[4]:
    st.subheader("Component allocation to HAL facilities")
    k = st.columns(3)
    k[0].metric("Transport cost", f"₹{tr['cost']:,.2f} lakh")
    k[1].metric("Units shipped", f"{tr['T'].sum():.0f}")
    k[2].metric("Spare supply", f"{sum(m.TR_SUPPLY) - tr['T'].sum():.0f}")
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(tr["T"], x=m.DESTS, y=m.ORIGINS, text_auto=".0f", color_continuous_scale="Blues",
                             title="Shipments (units)"))
    long = pd.DataFrame(tr["T"], index=m.ORIGINS, columns=m.DESTS).reset_index(names="Origin").melt("Origin")
    b.plotly_chart(px.bar(long, x="Origin", y="value", color="variable", title="Shipments by origin",
                          labels={"value": "Units", "variable": "Destination"}))
    st.caption("Costs are ₹ lakh per 100 units shipped; supply is treated as available capacity.")

# ---------------------------------------------------------------- 5 Assignment
with tabs[5]:
    st.subheader("Engineers to MRO jobs")
    k = st.columns(3)
    k[0].metric("Total processing time", f"{asg['total']} h")
    k[1].metric("Average per job", f"{asg['total'] / 8:.1f} h")
    k[2].metric("Fastest match", f"{m.AS_TIME[asg['rows'], asg['cols']].min()} h")
    fig = px.imshow(m.AS_TIME, x=m.JOBS, y=m.ENGINEERS, text_auto=True, color_continuous_scale="Blues_r",
                    title="Processing time (h) – red squares are the optimal assignment")
    fig.add_trace(go.Scatter(x=[m.JOBS[c] for c in asg["cols"]], y=[m.ENGINEERS[r] for r in asg["rows"]],
                             mode="markers", showlegend=False,
                             marker=dict(symbol="square-open", size=40, line=dict(width=3, color="red"))))
    st.plotly_chart(fig)
    st.dataframe(pd.DataFrame({"Engineer": [m.ENGINEERS[i] for i in asg["rows"]],
                               "Job": [m.JOBS[j] for j in asg["cols"]],
                               "Hours": m.AS_TIME[asg["rows"], asg["cols"]]}), hide_index=True)

# ---------------------------------------------------------------- 6 NLP
with tabs[6]:
    st.subheader("Capacity utilisation vs operating cost")
    st.latex(r"C(U) = 900 + 1800\,U + 2400\,U^2 \quad (\text{₹ crore}),\quad \text{Production} = 100\,U")
    mp = st.slider("Minimum production requirement (units)", 60, 100, 75)
    r = m.solve_nlp(mp)
    k = st.columns(3)
    k[0].metric("Optimal utilisation U", f"{r['u']:.2f}")
    k[1].metric("Production", f"{100 * r['u']:.0f} units")
    k[2].metric("Total cost", f"₹{r['cost']:,.0f} crore")
    u = np.linspace(0.6, 1, 81)
    fig = px.line(x=u, y=m.nlp_cost(u), labels={"x": "Utilisation U", "y": "Cost (₹ crore)"}, title="Total cost vs utilisation")
    fig.add_vrect(x0=0.6, x1=mp / 100, fillcolor="red", opacity=0.08, line_width=0, annotation_text="infeasible")
    fig.add_trace(go.Scatter(x=[r["u"]], y=[r["cost"]], mode="markers", marker=dict(size=14, color="red"), name="Optimum"))
    st.plotly_chart(fig)
    st.caption("Cost rises with U, so the optimum sits at the smallest feasible utilisation.")

# ---------------------------------------------------------------- 7 TSP
with tabs[7]:
    st.subheader("MRO / inspection service route")
    route = ts["route"]
    k = st.columns(3)
    k[0].metric("Total distance", f"{ts['total']:,} km")
    k[1].metric("Stops", 7)
    k[2].metric("Longest leg", f"{max(ts['legs']):,} km")
    lat = [m.NODE_LATLON[i][0] for i in route]
    lon = [m.NODE_LATLON[i][1] for i in route]
    fig = go.Figure(go.Scattergeo(lat=lat, lon=lon, mode="lines+markers", line=dict(width=2, color="#1f5fbf"),
                                  marker=dict(size=8)))
    fig.add_trace(go.Scattergeo(lat=[m.NODE_LATLON[i][0] for i in range(8)], lon=[m.NODE_LATLON[i][1] for i in range(8)],
                                text=m.NODES, mode="text", textposition="top right", showlegend=False))
    fig.update_geos(scope="asia", lataxis_range=[8, 30], lonaxis_range=[68, 90], showcountries=True)
    fig.update_layout(title="Optimal tour (map positions approximate)", showlegend=False, height=520)
    st.plotly_chart(fig)
    st.dataframe(pd.DataFrame({"Leg": range(1, 9), "From": [m.NODES[route[i]] for i in range(8)],
                               "To": [m.NODES[route[i + 1]] for i in range(8)], "km": ts["legs"]}), hide_index=True)
    st.caption("Exact optimum found by checking all 5,040 possible tours.")
