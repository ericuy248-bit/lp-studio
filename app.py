"""LP Studio: public, mobile-friendly two-variable LP graphical + pairing calculator.
Run: streamlit run app.py
"""
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="LP Studio | Graphical & Pairing Method",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = ["#3977e8", "#ec6474", "#8d62dc", "#00a68b", "#e4a044", "#9c6ac2", "#2c9dbb", "#d65d9a"]

st.markdown("""
<style>
    :root {--ink:#152340; --muted:#63718a; --line:#dce5f1; --navy:#152847; --blue:#3977e8;}
    .block-container {max-width: 1320px; padding-top: 1.1rem; padding-bottom: 2.5rem;}
    .lp-header {background:linear-gradient(112deg,#142747 4%,#224474 80%,#3262a6);border-radius:20px;padding:26px 30px;color:white;margin-bottom:1.25rem;box-shadow:0 8px 24px rgba(23,47,88,.12);}
    .lp-eyebrow {font-size:.72rem;font-weight:700;letter-spacing:.17em;text-transform:uppercase;color:#b8d7ff;margin-bottom:.35rem;}
    .lp-title {font-size:clamp(1.8rem,3.5vw,2.7rem);font-weight:800;line-height:1.1;margin-bottom:.35rem;}
    .lp-sub {font-size:.95rem;color:#dceaff;line-height:1.6;}
    .lp-heading {font-size:1.22rem;font-weight:750;color:#152340;margin-top:.35rem;margin-bottom:.35rem;}
    .lp-note {color:#65758c;font-size:.89rem;line-height:1.5;}
    .lp-tag {display:inline-block;border-radius:24px;padding:4px 11px;font-size:.78rem;font-weight:700;letter-spacing:.04em;margin-left:4px;background:rgba(255,255,255,.16);color:#fff;}
    .lp-kicker {font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;font-weight:700;color:#70809a;margin-bottom:.4rem;}
    .lp-rule {height:1px;background:#dfe6ef;margin:1rem 0;}
    .lp-pairbox {border-radius:12px;background:#f1f6fd;padding:15px 19px;margin-bottom:.8rem;border:1px solid #dce8f6;}
    .lp-pairbox strong {color:#152b4d;}
    [data-testid="stMetric"] {border:1px solid #e0e8f2;background:#fff;border-radius:12px;padding:12px 14px;}
    [data-testid="stMetricLabel"] {color:#64748b;}
    div[data-testid="stTabs"] button {font-weight:650;}
    div[data-testid="stDataFrame"] {border-radius:12px;overflow:hidden;}
    @media(max-width:640px){.lp-header{padding:21px 20px;border-radius:14px}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""", unsafe_allow_html=True)


def fraction(text):
    try:
        return Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"'{text}' is not a valid number. Use 4, -1.5, or 3/2.") from exc


def fmt(x):
    return str(x)


def approximate(x):
    return f"{float(x):,.5f}".rstrip("0").rstrip(".")


def eq_text(c, boundary=False):
    symbol = "=" if boundary else c["op"]
    return f"({fmt(c['a'])})x₁ + ({fmt(c['b'])})x₂ {symbol} {fmt(c['r'])}"


def satisfied(c, p):
    left = c["a"] * p[0] + c["b"] * p[1]
    if c["op"] == "≤":
        return left <= c["r"]
    if c["op"] == "≥":
        return left >= c["r"]
    return left == c["r"]


def intersect(c, d):
    a, b, r = c["a"], c["b"], c["r"]
    e, f, s = d["a"], d["b"], d["r"]
    if (a == 0 and b == 0) or (e == 0 and f == 0):
        return None, "Constant constraint, not a boundary line"
    D = a * f - e * b
    if D == 0:
        same = (a*s == e*r and b*s == f*r)
        return None, "Coincident boundaries" if same else "Parallel boundaries"
    Dx = r*f - s*b
    Dy = a*s - e*r
    return (Dx/D, Dy/D), (D, Dx, Dy)


def feasible_direction(constraints, direction):
    dx, dy = direction
    if dx == 0 and dy == 0:
        return False
    for c in constraints:
        expr = c["a"] * dx + c["b"] * dy
        if c["op"] == "≤" and expr > 0:
            return False
        if c["op"] == "≥" and expr < 0:
            return False
        if c["op"] == "=" and expr != 0:
            return False
    return True


def solve(original, c1, c2, mode):
    # By default, non-negativity constraints are part of the model.
    axes = [
        {"name": "x₁ ≥ 0", "a": Fraction(1), "b": Fraction(0), "op": "≥", "r": Fraction(0)},
        {"name": "x₂ ≥ 0", "a": Fraction(0), "b": Fraction(1), "op": "≥", "r": Fraction(0)},
    ]
    constraints = original + axes
    pairs = []
    points = {}
    feasible = {}
    for a, b in combinations(constraints, 2):
        p, working = intersect(a, b)
        valid = p is not None and all(satisfied(c, p) for c in constraints)
        pairs.append({"first": a, "second": b, "point": p, "working": working, "valid": valid})
        if p is not None:
            points[p] = points.get(p, False) or valid
            if valid:
                feasible[p] = c1 * p[0] + c2 * p[1]

    ray_candidates = [(Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    for c in constraints:
        ray_candidates.extend([(c["b"], -c["a"]), (-c["b"], c["a"])])
    rays = [r for r in dict.fromkeys(ray_candidates) if feasible_direction(constraints, r)]
    sign = 1 if mode == "Maximize" else -1
    improving_ray = next((r for r in rays if sign*(c1*r[0] + c2*r[1]) > 0), None)

    if not feasible:
        status, optimum, best = "INFEASIBLE", None, []
    elif improving_ray is not None:
        status, optimum, best = "UNBOUNDED", None, []
    else:
        optimum = (max if mode == "Maximize" else min)(feasible.values())
        best = [p for p, z in feasible.items() if z == optimum]
        status = "OPTIMAL"

    alternate = status == "OPTIMAL" and (len(best) > 1 or any(c1*r[0] + c2*r[1] == 0 for r in rays))
    return {
        "original": original, "constraints": constraints, "pairs": pairs, "points": points,
        "feasible": feasible, "status": status, "optimum": optimum, "best": best,
        "alternate": alternate, "improving_ray": improving_ray,
        "mode": mode, "c1": c1, "c2": c2,
    }


def clip_polygon(poly, a, b, rhs):
    """Clip polygon by a*x+b*y <= rhs; used for DISPLAY only."""
    if not poly:
        return []
    out = []
    for i in range(len(poly)):
        p, q = poly[i-1], poly[i]
        vp, vq = a*p[0]+b*p[1]-rhs, a*q[0]+b*q[1]-rhs
        tp, tq = vp <= 1e-9, vq <= 1e-9
        if tp != tq and abs(vp - vq) > 1e-15:
            t = vp / (vp-vq)
            out.append((p[0]+t*(q[0]-p[0]), p[1]+t*(q[1]-p[1])))
        if tq:
            out.append(q)
    return out


def graph_limits(solution):
    xvals = [8.0]
    yvals = [8.0]
    for p in solution["points"]:
        if p[0] >= 0 and p[1] >= 0:
            xvals.append(float(p[0]))
            yvals.append(float(p[1]))
    for c in solution["original"]:
        if c["a"] and c["r"]/c["a"] > 0:
            xvals.append(float(c["r"]/c["a"]))
        if c["b"] and c["r"]/c["b"] > 0:
            yvals.append(float(c["r"]/c["b"]))
    return max(xvals)*1.18, max(yvals)*1.18


def boundary_segment(c, xmax, ymax):
    a, b, r = float(c["a"]), float(c["b"]), float(c["r"])
    if a == 0 and b == 0:
        return []
    candidates = []
    if b != 0:
        candidates.extend([(0, r/b), (xmax, (r-a*xmax)/b)])
    if a != 0:
        candidates.extend([(r/a, 0), ((r-b*ymax)/a, ymax)])
    result = []
    for x, y in candidates:
        if -1e-8 <= x <= xmax+1e-8 and -1e-8 <= y <= ymax+1e-8:
            if not any(abs(px-x)+abs(py-y) < 1e-7 for px, py in result):
                result.append((min(max(x,0),xmax),min(max(y,0),ymax)))
    return result[:2]


def make_figure(sol, show_invalid=True):
    xmax, ymax = graph_limits(sol)
    fig = go.Figure()
    poly = [(0.,0.),(xmax,0.),(xmax,ymax),(0.,ymax)]
    for c in sol["constraints"]:
        a, b, rhs = float(c["a"]), float(c["b"]), float(c["r"])
        if c["op"] == "≤":
            pieces = [(a,b,rhs)]
        elif c["op"] == "≥":
            pieces = [(-a,-b,-rhs)]
        else:
            pieces = [(a,b,rhs),(-a,-b,-rhs)]
        for aa, bb, rr in pieces:
            poly = clip_polygon(poly,aa,bb,rr)
    if len(poly) >= 3:
        vertices = poly + [poly[0]]
        fig.add_trace(go.Scatter(x=[p[0] for p in vertices], y=[p[1] for p in vertices],
            fill="toself",fillcolor="rgba(24,174,120,.15)",line=dict(color="rgba(25,167,110,.3)",width=1),
            name="Feasible region",hoverinfo="skip"))
    elif len(poly) == 2:
        fig.add_trace(go.Scatter(x=[p[0] for p in poly],y=[p[1] for p in poly],mode="lines",
            line=dict(color="#11aa80",width=7),name="Feasible set",hoverinfo="skip"))
    elif len(poly) == 1:
        fig.add_trace(go.Scatter(x=[poly[0][0]],y=[poly[0][1]],mode="markers",
            marker=dict(color="#11aa80",size=11),name="Feasible set"))
    for i, c in enumerate(sol["original"]):
        segment = boundary_segment(c,xmax,ymax)
        if len(segment) == 2:
            fig.add_trace(go.Scatter(x=[p[0] for p in segment],y=[p[1] for p in segment],mode="lines",
                line=dict(color=COLORS[i % len(COLORS)],width=2.8),name=c["name"],
                hovertemplate=f"{c['name']}: {eq_text(c)}<extra></extra>"))
    for kind, pts, color, symbol, size in [
        ("Invalid intersections",[p for p, valid in sol["points"].items() if not valid],"#9aa4b6","x",9),
        ("Feasible corners",[p for p in sol["feasible"] if p not in sol["best"]],"#029a6f","circle",10),
        ("Optimal corner(s)",sol["best"],"#f59f25","circle",13),
    ]:
        if kind == "Invalid intersections" and not show_invalid:
            continue
        pts = [p for p in pts if 0 <= float(p[0]) <= xmax and 0 <= float(p[1]) <= ymax]
        if pts:
            fig.add_trace(go.Scatter(x=[float(p[0]) for p in pts],y=[float(p[1]) for p in pts],mode="markers+text",
                text=[f"({fmt(p[0])}, {fmt(p[1])})" if kind != "Invalid intersections" else "" for p in pts],
                textposition="top center",textfont=dict(size=11,color="#21344d"),
                marker=dict(color=color,size=size,symbol=symbol,line=dict(color="white",width=1.5)),
                name=kind,customdata=[f"({fmt(p[0])}, {fmt(p[1])})" for p in pts],
                hovertemplate="Point %{customdata}<extra>"+kind+"</extra>"))
    fig.update_layout(template="plotly_white",height=520,margin=dict(l=12,r=15,t=20,b=35),
        paper_bgcolor="#fff",plot_bgcolor="#fbfdff",hovermode="closest",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(size=11)),
        xaxis=dict(title="x₁",range=[0,xmax],zeroline=True,zerolinewidth=2,zerolinecolor="#495b75",
                   gridcolor="#e7edf5",ticks="outside"),
        yaxis=dict(title="x₂",range=[0,ymax],zeroline=True,zerolinewidth=2,zerolinecolor="#495b75",
                   gridcolor="#e7edf5",ticks="outside"),
        font=dict(family="Arial",color="#21344d"),
    )
    return fig


def pairing_rows(sol):
    out = []
    for i,p in enumerate(sol["pairs"],1):
        point=p["point"]
        out.append({
            "No.":i,
            "Pair": f"{p['first']['name']} × {p['second']['name']}",
            "Intersection":f"({fmt(point[0])}, {fmt(point[1])})" if point else "—",
            "Verdict":"✅ VALID" if p["valid"] else ("❌ INVALID" if point else "— NO UNIQUE POINT"),
            "Z":fmt(sol["feasible"][point]) if p["valid"] else "—",
        })
    return out


def report_text(sol):
    lines=["LP STUDIO — COMPLETE PAIRING METHOD REPORT", "="*70,
        f"{sol['mode'].upper()} Z = {fmt(sol['c1'])}x1 + ({fmt(sol['c2'])})x2", "SUBJECT TO:"]
    lines.extend(f"  {c['name']}: {eq_text(c)}" for c in sol["constraints"])
    lines.extend(["", "CRAMER'S RULE: D=a1*b2-a2*b1; Dx=r1*b2-r2*b1; Dy=a1*r2-a2*r1", ""])
    for i,p in enumerate(sol["pairs"],1):
        a,b=p["first"],p["second"]
        lines.extend([f"PAIR {i}: {a['name']} × {b['name']}",
            f"  {eq_text(a,boundary=True)}", f"  {eq_text(b,boundary=True)}"])
        if p["point"] is None:
            lines.append(f"  No unique solution: {p['working']}")
        else:
            D,Dx,Dy=p["working"]
            x,y=p["point"]
            lines.extend([f"  D = ({a['a']})({b['b']}) - ({b['a']})({a['b']}) = {D}",
                f"  Dx = ({a['r']})({b['b']}) - ({b['r']})({a['b']}) = {Dx}",
                f"  Dy = ({a['a']})({b['r']}) - ({b['a']})({a['r']}) = {Dy}",
                f"  x1=Dx/D={x}; x2=Dy/D={y}","  CONSTRAINT CHECKS:"])
            for c in sol["constraints"]:
                lhs=c["a"]*x+c["b"]*y
                lines.append(f"    {c['name']}: {c['a']}({x}) + {c['b']}({y}) = {lhs} {c['op']} {c['r']} -> {'PASS' if satisfied(c,(x,y)) else 'FAIL'}")
            lines.append(f"  VERDICT: {'VALID' if p['valid'] else 'INVALID'}")
            if p["valid"]:
                lines.append(f"  Z = {sol['c1']}({x}) + {sol['c2']}({y}) = {sol['feasible'][(x,y)]}")
        lines.extend(["-"*70,""])
    lines.append("FEASIBLE CORNER COMPARISON:")
    for point,z in sorted(sol["feasible"].items()):
        lines.append(f"  ({point[0]}, {point[1]}) -> Z = {z}")
    lines.append("RESULT: "+sol["status"])
    if sol["status"]=="OPTIMAL":
        lines.append("Optimal Z: "+str(sol["optimum"]))
        for p in sol["best"]:
            lines.append(f"Optimal corner: x1={p[0]}, x2={p[1]}")
        if sol["alternate"]:
            lines.append("Multiple optimal feasible solutions exist.")
    elif sol["status"]=="UNBOUNDED":
        lines.append("No finite optimum; improving ray: "+str(sol["improving_ray"]))
    else:
        lines.append("No feasible points.")
    return "\n".join(lines)


# ----- INTERACTIVE APP -----
DEFAULT_ROWS = [("6", "5", "≤", "69"), ("4", "7", "≤", "77"), ("8", "1", "≤", "74")]


def initial_state():
    defaults={"obj_sense":"Maximize","obj_c1":"13","obj_c2":"12","constraint_count":3}
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v
    for i in range(12):
        row=DEFAULT_ROWS[i] if i<3 else ("0","0","≤","0")
        for suffix, value in zip(("a","b","op","rhs"),row):
            key=f"constraint_{suffix}_{i}"
            if key not in st.session_state:
                st.session_state[key]=value


def reset_example():
    st.session_state["obj_sense"]="Maximize"
    st.session_state["obj_c1"]="13"
    st.session_state["obj_c2"]="12"
    st.session_state["constraint_count"]=3
    for i,row in enumerate(DEFAULT_ROWS):
        for suffix, value in zip(("a","b","op","rhs"),row):
            st.session_state[f"constraint_{suffix}_{i}"]=value


initial_state()
st.markdown("""
<div class="lp-header">
    <div class="lp-eyebrow">Operations Research · Interactive Learning Tool</div>
    <div class="lp-title">LP Studio <span class="lp-tag">2 VARIABLES</span></div>
    <div class="lp-sub">Explore linear programming with shaded graphs, exact intersection points, Cramer's Rule, and every feasibility check.</div>
</div>
""", unsafe_allow_html=True)

heading, resetcol = st.columns([4,1])
with heading:
    st.markdown('<div class="lp-heading">Build your LP problem</div>',unsafe_allow_html=True)
    st.caption("Numbers can be integers, decimals, or fractions (for example 3/2).")
with resetcol:
    st.button("↺ Load example",use_container_width=True,on_click=reset_example)

with st.expander("✏️ Edit objective function and constraints", expanded=False):
    st.caption("Open this panel to change the objective, number of constraints, coefficients, or inequality signs. Results refresh automatically.")
    with st.container(border=True):
        oc1,oc2,oc3 = st.columns([1.15,1,1])
        with oc1:
            mode = st.selectbox("Optimization",["Maximize","Minimize"],key="obj_sense")
        with oc2:
            c1txt = st.text_input("Coefficient of x₁",key="obj_c1")
        with oc3:
            c2txt = st.text_input("Coefficient of x₂",key="obj_c2")
        st.markdown(f"**Objective:** {mode}  `Z = ({c1txt})x₁ + ({c2txt})x₂`")
        count = st.number_input("Number of constraints",min_value=1,max_value=12,step=1,key="constraint_count")
        st.caption("Choose ≤, ≥, or = for each constraint. Nonnegativity (x₁ ≥ 0, x₂ ≥ 0) is included automatically.")
        for i in range(int(count)):
            with st.container(border=True):
                st.markdown(f"**Constraint C{i+1}**")
                ac,bc,opc,rc=st.columns([1,1,0.9,1.05])
                ac.text_input("x₁",key=f"constraint_a_{i}")
                bc.text_input("x₂",key=f"constraint_b_{i}")
                opc.selectbox("Relation",["≤","≥","="],key=f"constraint_op_{i}")
                rc.text_input("Right side",key=f"constraint_rhs_{i}")

try:
    c1,c2=fraction(c1txt),fraction(c2txt)
    original=[]
    for i in range(int(count)):
        original.append({
            "name":f"C{i+1}",
            "a":fraction(st.session_state[f"constraint_a_{i}"]),
            "b":fraction(st.session_state[f"constraint_b_{i}"]),
            "op":st.session_state[f"constraint_op_{i}"],
            "r":fraction(st.session_state[f"constraint_rhs_{i}"]),
        })
    sol=solve(original,c1,c2,mode)
except ValueError as exc:
    st.error(f"Correct your inputs to continue: {exc}")
    st.stop()

st.markdown('<div class="lp-heading">Solution dashboard</div>',unsafe_allow_html=True)
if sol["status"]=="OPTIMAL":
    status="✅ OPTIMAL"
    result_text=f"Z = {fmt(sol['optimum'])} ≈ {approximate(sol['optimum'])}"
    point_text="; ".join(f"({fmt(p[0])}, {fmt(p[1])})" for p in sol["best"])
    if sol["alternate"]:
        point_text += " · Alternative optimal points exist"
elif sol["status"]=="UNBOUNDED":
    status="↗️ UNBOUNDED"
    result_text="No finite optimum"
    point_text="Objective improves indefinitely along a feasible direction"
else:
    status="❌ INFEASIBLE"
    result_text="No feasible solution"
    point_text="No point satisfies every constraint"

scol1,scol2,scol3=st.columns([1.15,1.2,1.4])
scol1.metric("Solution status",status)
scol2.metric("Objective value",fmt(sol["optimum"]) if sol["optimum"] is not None else "—")
scol3.metric("Valid corner points",len(sol["feasible"]))
st.caption(point_text)

graph_tab,pair_tab,summary_tab=st.tabs(["📈 Graphical method","🧮 Pairing method","📋 Final comparison"])

with graph_tab:
    st.markdown('<div class="lp-heading">Graphical solution</div>',unsafe_allow_html=True)
    st.markdown('<div class="lp-note">The green region satisfies all constraints. Green points are valid corners; orange points are optimal. Zoom and pan directly on the graph.</div>',unsafe_allow_html=True)
    show_invalid=st.checkbox("Show invalid intersection points",value=True)
    st.plotly_chart(make_figure(sol,show_invalid),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
    if sol["status"]=="UNBOUNDED":
        st.warning("The graph shows only a finite viewing window. The LP has no finite optimal objective value.")
    elif sol["status"]=="INFEASIBLE":
        st.error("There is no common feasible region. Some drawn lines may still intersect outside the permitted region.")
    elif sol["alternate"]:
        st.info("There are multiple optimal solutions. Highlighted points show optimal corners, but other points between/along them may also be optimal.")
    st.caption("Graph coordinates are displayed approximately; all mathematical decisions and substitutions use exact fractions.")

with pair_tab:
    st.markdown('<div class="lp-heading">Constraint pairing · step by step</div>',unsafe_allow_html=True)
    st.markdown('<div class="lp-note">Every constraint is paired with every other boundary, including x₁=0 and x₂=0. Select a pair to see exactly how x₁ and x₂ were calculated and tested.</div>',unsafe_allow_html=True)

    rows=pairing_rows(sol)
    filtered_mode=st.radio("Filter pairs",["All pairs","Valid only","Invalid only","No unique intersection"],horizontal=True)
    if filtered_mode=="Valid only":
        inds=[i for i,p in enumerate(sol["pairs"]) if p["valid"]]
    elif filtered_mode=="Invalid only":
        inds=[i for i,p in enumerate(sol["pairs"]) if p["point"] is not None and not p["valid"]]
    elif filtered_mode=="No unique intersection":
        inds=[i for i,p in enumerate(sol["pairs"]) if p["point"] is None]
    else:
        inds=list(range(len(rows)))
    if inds:
        st.dataframe(pd.DataFrame([rows[i] for i in inds]),hide_index=True,use_container_width=True,
            height=min(370,85+len(inds)*36))
        choice=st.selectbox("Inspect one pair",options=inds,
            format_func=lambda i:f"Pair {i+1:02d}  ·  {rows[i]['Pair']}  ·  {rows[i]['Verdict']}")
        pair=sol["pairs"][choice]
        a,b=pair["first"],pair["second"]
        st.markdown(f"<div class='lp-pairbox'><strong>Pair {choice+1:02d}</strong> &nbsp; {a['name']} × {b['name']}<br><span style='color:#596b84'>Replace the inequality symbols with equality to find a candidate intersection.</span></div>",unsafe_allow_html=True)
        st.markdown("##### 1 · Write the two boundary equations")
        e1,e2=st.columns(2)
        e1.code(eq_text(a,boundary=True),language=None)
        e2.code(eq_text(b,boundary=True),language=None)
        st.markdown("##### 2 · Solve using Cramer's Rule")
        if pair["point"] is None:
            st.warning(f"There is no unique intersection: {pair['working']}.")
        else:
            D,Dx,Dy=pair["working"]
            x,y=pair["point"]
            st.latex(r"D=a_1b_2-a_2b_1\qquad D_x=r_1b_2-r_2b_1\qquad D_y=a_1r_2-a_2r_1")
            d1,d2,d3=st.columns(3)
            d1.metric("D",fmt(D))
            d2.metric("Dₓ",fmt(Dx))
            d3.metric("Dᵧ",fmt(Dy))
            st.code(f"D  = ({a['a']})({b['b']}) - ({b['a']})({a['b']}) = {D}\n"
                    f"Dx = ({a['r']})({b['b']}) - ({b['r']})({a['b']}) = {Dx}\n"
                    f"Dy = ({a['a']})({b['r']}) - ({b['a']})({a['r']}) = {Dy}\n"
                    f"x1 = Dx / D = {Dx} / ({D}) = {x}\n"
                    f"x2 = Dy / D = {Dy} / ({D}) = {y}",language="text")
            cpx,cpy=st.columns(2)
            cpx.metric("Intersection x₁",fmt(x),help=f"≈ {approximate(x)}")
            cpy.metric("Intersection x₂",fmt(y),help=f"≈ {approximate(y)}")
            st.markdown("##### 3 · Substitute into **every** constraint")
            checks=[]
            for c in sol["constraints"]:
                lhs=c["a"]*x+c["b"]*y
                checks.append({"Constraint":c["name"],
                    "Substitution":f"({c['a']})({x}) + ({c['b']})({y})",
                    "Left side":fmt(lhs),"Relation":c["op"],"Right side":fmt(c["r"]),
                    "Result":"✅ PASS" if satisfied(c,(x,y)) else "❌ FAIL"})
            st.dataframe(pd.DataFrame(checks),hide_index=True,use_container_width=True)
            st.markdown("##### 4 · Feasibility verdict")
            if pair["valid"]:
                z=sol["feasible"][(x,y)]
                st.success(f"VALID CORNER: ({fmt(x)}, {fmt(y)}) satisfies every constraint.")
                st.latex(rf"Z=({fmt(c1)})({fmt(x)})+({fmt(c2)})({fmt(y)})={fmt(z)}")
            else:
                fails=[c["name"] for c in sol["constraints"] if not satisfied(c,(x,y))]
                st.error(f"INVALID POINT: ({fmt(x)}, {fmt(y)}) fails {', '.join(fails)} and must not be considered for the optimum.")
    else:
        st.info("No pairs match this filter. Choose another filter to inspect the calculations.")
    st.download_button("⬇ Download complete step-by-step solution",report_text(sol),
        file_name="LP_Studio_Pairing_Solution.txt",mime="text/plain",use_container_width=True)

with summary_tab:
    st.markdown('<div class="lp-heading">Compare the feasible corner points</div>',unsafe_allow_html=True)
    corners=sorted(sol["feasible"].items(),key=lambda row:(row[0][0],row[0][1]))
    if corners:
        st.dataframe(pd.DataFrame([{"x₁":fmt(p[0]),"x₂":fmt(p[1]),"Z":fmt(z),
                "Approx. Z":approximate(z),"Result":"⭐ OPTIMAL CORNER" if p in sol["best"] else "Feasible corner"}
                for p,z in corners]),hide_index=True,use_container_width=True)
    else:
        st.info("There are no feasible corner points.")
    if sol["status"]=="OPTIMAL":
        st.success(result_text)
        for p in sol["best"]:
            st.write(f"**Optimal corner:** x₁ = {fmt(p[0])}, x₂ = {fmt(p[1])}")
        if sol["alternate"]:
            st.info("More than one feasible point gives the same optimal objective value.")
    elif sol["status"]=="UNBOUNDED":
        st.warning(f"The {sol['mode'].lower()} problem has no finite optimum. Improving feasible direction: {sol['improving_ray']}.")
    else:
        st.error("The constraints are inconsistent; no feasible solution exists.")
    st.markdown('<div class="lp-rule"></div>',unsafe_allow_html=True)
    st.caption("This app solves continuous linear programs with TWO decision variables. It does not enforce integer decisions or perform branch-and-bound. Graphs are visual aids; classifications use exact arithmetic.")
