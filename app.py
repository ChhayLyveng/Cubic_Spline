import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────
# Core algorithms (from notebook)
# ─────────────────────────────────────────────

def thomas_algorithm(a, b, c, d):
    """Tridiagonal (Thomas) solver.  a=lower, b=main, c=upper, d=RHS."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)
    n = len(b)
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def spline_second_derivatives(x, y, clamped=False, fp0=None, fpn=None):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1
    h = np.diff(x)
    lower, diag, upper, rhs = [], [], [], []

    if clamped:
        diag.append(2 * h[0])
        upper.append(h[0])
        rhs.append(6 * ((y[1] - y[0]) / h[0] - fp0))
    else:
        diag.append(1); upper.append(0); rhs.append(0)

    for i in range(1, n):
        lower.append(h[i - 1])
        diag.append(2 * (h[i - 1] + h[i]))
        upper.append(h[i])
        rhs.append(6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]))

    if clamped:
        lower.append(h[-1])
        diag.append(2 * h[-1])
        rhs.append(6 * (fpn - (y[-1] - y[-2]) / h[-1]))
    else:
        lower.append(0); diag.append(1); rhs.append(0)

    upper = upper[:len(diag) - 1]
    lower = lower[:len(diag) - 1]
    return thomas_algorithm(lower, diag, upper, rhs)


def evaluate_spline(x, y, M, xq):
    x = np.array(x); y = np.array(y)
    vals = []
    for xv in np.array(xq):
        i = np.searchsorted(x, xv) - 1
        i = max(0, min(i, len(x) - 2))
        h = x[i + 1] - x[i]
        A = (x[i + 1] - xv) / h
        B = (xv - x[i]) / h
        s = A * y[i] + B * y[i + 1] + ((A**3 - A) * M[i] + (B**3 - B) * M[i + 1]) * h**2 / 6
        vals.append(s)
    return np.array(vals)


def segment_coefficients(x, y, M):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float); M = np.array(M, dtype=float)
    h = np.diff(x)
    coeffs = []
    for i in range(len(h)):
        hi = h[i]
        a = y[i]
        b = (y[i + 1] - y[i]) / hi - hi * (2 * M[i] + M[i + 1]) / 6
        c = M[i] / 2
        d = (M[i + 1] - M[i]) / (6 * hi)
        coeffs.append((a, b, c, d))
    return coeffs


# ─────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────

st.set_page_config(page_title="Cubic Spline Interpolation", layout="wide")

st.title("🔵 Cubic Spline Interpolation")
st.markdown(
    "Interactive tool based on the **I3-AMS-A Group 3** mini-project. "
    "Use the tabs below to interpolate data with splines, evaluate at custom points, "
    "or solve a tridiagonal system."
)

tab_spline, tab_thomas = st.tabs(["📈 Cubic Spline", "🔢 Thomas Algorithm (AX = B)"])


# ════════════════════════════════════════════
# TAB 1 – Cubic Spline
# ════════════════════════════════════════════
with tab_spline:

    col_in, col_out = st.columns([1, 1.6], gap="large")

    with col_in:
        st.subheader("Input Data Points")

        # ── preset / manual toggle ──
        use_preset = st.checkbox("Use preset example (curve 1 from notebook)", value=True)

        if use_preset:
            default_x = "1, 2, 5, 6, 7, 8, 10, 13, 17"
            default_y = "3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5"
        else:
            default_x = ""
            default_y = ""

        x_raw = st.text_input("x values (comma-separated)", value=default_x)
        y_raw = st.text_input("y values (comma-separated)", value=default_y)

        # ── spline type ──
        st.subheader("Spline Type")
        spline_type = st.radio("", ["Natural", "Clamped"], horizontal=True)

        fp0 = fpn = None
        if spline_type == "Clamped":
            c1, c2 = st.columns(2)
            fp0 = c1.number_input("f ′(x₀)  – left endpoint derivative", value=1.0, step=0.1)
            fpn = c2.number_input("f ′(xₙ)  – right endpoint derivative", value=-0.67, step=0.01)

        # ── query point ──
        st.subheader("Evaluate at a Point")
        show_query = st.checkbox("Evaluate spline at custom x", value=True)
        query_x = None
        if show_query:
            query_x = st.number_input("Query x value", value=4.0, step=0.5)

        # ── comparison toggle ──
        show_both = st.checkbox("Overlay Natural AND Clamped curves for comparison", value=True)
        if show_both and spline_type == "Clamped":
            st.caption("When comparing, clamped uses the derivatives entered above.")
        elif show_both:
            c3, c4 = st.columns(2)
            fp0_comp = c3.number_input("Clamped f ′(x₀) (for comparison)", value=1.0, step=0.1)
            fpn_comp = c4.number_input("Clamped f ′(xₙ) (for comparison)", value=-0.67, step=0.01)

        run = st.button("▶  Compute & Plot", type="primary", use_container_width=True)

    # ── right column: results ──
    with col_out:
        if run:
            try:
                xs = [float(v.strip()) for v in x_raw.split(",") if v.strip()]
                ys = [float(v.strip()) for v in y_raw.split(",") if v.strip()]

                if len(xs) != len(ys):
                    st.error("x and y must have the same number of values.")
                elif len(xs) < 3:
                    st.error("At least 3 data points required.")
                elif xs != sorted(xs):
                    st.error("x values must be strictly increasing.")
                else:
                    # ── compute splines ──
                    if spline_type == "Natural":
                        M = spline_second_derivatives(xs, ys)
                    else:
                        M = spline_second_derivatives(xs, ys, clamped=True, fp0=fp0, fpn=fpn)

                    # comparison
                    M_nat = spline_second_derivatives(xs, ys)
                    if show_both:
                        if spline_type == "Clamped":
                            M_clamp = M
                        else:
                            M_clamp = spline_second_derivatives(
                                xs, ys, clamped=True,
                                fp0=fp0_comp, fpn=fpn_comp
                            )

                    # ── PLOT ──
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    x_fine = np.linspace(min(xs), max(xs), 400)

                    if show_both:
                        yn = evaluate_spline(xs, ys, M_nat, x_fine)
                        yc = evaluate_spline(xs, ys, M_clamp, x_fine)
                        ax.plot(x_fine, yn, color="#2196F3", linewidth=2, label="Natural spline")
                        ax.plot(x_fine, yc, "--", color="#FF5722", linewidth=2, label="Clamped spline")
                    else:
                        y_curve = evaluate_spline(xs, ys, M, x_fine)
                        color = "#2196F3" if spline_type == "Natural" else "#FF5722"
                        ax.plot(x_fine, y_curve, color=color, linewidth=2, label=f"{spline_type} spline")

                    ax.scatter(xs, ys, color="#333", zorder=5, s=55, label="Data points")

                    if show_query and query_x is not None:
                        qy = evaluate_spline(xs, ys, M, [query_x])[0]
                        ax.axvline(query_x, color="gray", linestyle=":", linewidth=1)
                        ax.scatter([query_x], [qy], color="red", zorder=6, s=90,
                                   label=f"Query  ({query_x}, {qy:.4f})")

                    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
                    ax.set_title("Cubic Spline Interpolation")
                    ax.legend(); ax.grid(True, alpha=0.3)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # ── query result ──
                    if show_query and query_x is not None:
                        qy = evaluate_spline(xs, ys, M, [query_x])[0]
                        st.success(f"**f({query_x}) ≈ {qy:.6f}**  ({spline_type} spline)")

                    # ── second derivatives ──
                    with st.expander("📋 Second Derivatives  M = f ″(xᵢ)"):
                        df_M = pd.DataFrame({
                            "i": range(len(xs)),
                            "xᵢ": xs,
                            "yᵢ": ys,
                            "M_natural": np.round(M_nat, 8),
                            f"M_{spline_type.lower()}": np.round(M, 8),
                        })
                        st.dataframe(df_M, use_container_width=True, hide_index=True)

                    # ── segment coefficients ──
                    with st.expander("📐 Segment Coefficients  s(x) = a + b·(x−xᵢ) + c·(x−xᵢ)² + d·(x−xᵢ)³"):
                        coeffs = segment_coefficients(xs, ys, M)
                        rows = []
                        for j, (a, b, c, d) in enumerate(coeffs):
                            rows.append({
                                "seg": j,
                                f"[x{j}, x{j+1}]": f"[{xs[j]}, {xs[j+1]}]",
                                "a": round(a, 8),
                                "b": round(b, 8),
                                "c": round(c, 8),
                                "d": round(d, 8),
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # ── node table ──
                    with st.expander("📊 Node values  xᵢ, f(xᵢ), f ′(xᵢ)"):
                        h_arr = np.diff(xs)
                        rows2 = []
                        for i, xi in enumerate(xs):
                            fi = ys[i]
                            if i < len(xs) - 1:
                                fpi = (ys[i + 1] - ys[i]) / h_arr[i] - h_arr[i] * (2 * M[i] + M[i + 1]) / 6
                            else:
                                fpi = None
                            rows2.append({
                                "i": i,
                                "xᵢ": round(xi, 6),
                                "f(xᵢ)": round(fi, 6),
                                "f′(xᵢ)": round(fpi, 6) if fpi is not None else "—",
                            })
                        st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

            except ValueError as e:
                st.error(f"Parse error: {e}. Make sure all values are numbers separated by commas.")
        else:
            st.info("👈  Configure inputs on the left and click **▶ Compute & Plot**.")


# ════════════════════════════════════════════
# TAB 2 – Thomas Algorithm
# ════════════════════════════════════════════
with tab_thomas:
    st.subheader("Solve a Tridiagonal System  AX = B")
    st.markdown(
        "Enter the three diagonals and the right-hand side vector as comma-separated values. "
        "For an **n × n** matrix: main diagonal has **n** entries, "
        "upper and lower diagonals have **n − 1** entries each."
    )

    col_t1, col_t2 = st.columns([1, 1.2], gap="large")

    with col_t1:
        t_lower = st.text_input("Lower diagonal (a)", value="1, 1, 1, 1", key="tl")
        t_diag  = st.text_input("Main diagonal (b)",  value="2, 2, 2, 2, 2", key="td")
        t_upper = st.text_input("Upper diagonal (c)", value="1, 1, 1, 1", key="tu")
        t_rhs   = st.text_input("RHS vector (d)",     value="1, 1, 4, 6, 5", key="tr")
        solve_btn = st.button("▶  Solve", type="primary", use_container_width=True)

    with col_t2:
        if solve_btn:
            try:
                a = [float(v.strip()) for v in t_lower.split(",") if v.strip()]
                b = [float(v.strip()) for v in t_diag.split(",") if v.strip()]
                c = [float(v.strip()) for v in t_upper.split(",") if v.strip()]
                d = [float(v.strip()) for v in t_rhs.split(",") if v.strip()]
                n = len(b)

                if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
                    st.error(
                        f"Size mismatch. For n={n}: "
                        f"lower needs {n-1} values (got {len(a)}), "
                        f"upper needs {n-1} values (got {len(c)}), "
                        f"RHS needs {n} values (got {len(d)})."
                    )
                else:
                    sol = thomas_algorithm(a, b, c, d)

                    st.success("**Solution X:**")
                    df_sol = pd.DataFrame({
                        "Index i": range(n),
                        "xᵢ": np.round(sol, 8),
                    })
                    st.dataframe(df_sol, use_container_width=True, hide_index=True)

                    # verify AX = B
                    A = np.diag(b, 0) + np.diag(a, -1) + np.diag(c, 1)
                    residual = np.linalg.norm(A @ sol - d)
                    st.caption(f"Residual ‖AX − B‖ = {residual:.2e}")

                    # bar chart
                    fig2, ax2 = plt.subplots(figsize=(6, 3))
                    ax2.bar(range(n), sol, color="#2196F3", edgecolor="white")
                    ax2.set_xlabel("Index i"); ax2.set_ylabel("xᵢ")
                    ax2.set_title("Solution vector X"); ax2.grid(True, alpha=0.3, axis="y")
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

            except ValueError as e:
                st.error(f"Parse error: {e}")
        else:
            st.info("Enter the diagonals and click **▶ Solve**.")

st.markdown("---")
st.caption("Mini Project – Cubic Spline | I3-AMS-A Group 3 | Built with Streamlit")