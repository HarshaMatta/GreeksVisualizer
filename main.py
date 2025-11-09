import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# --- Core Mathematical Functions ---

def norm_cdf(x):
    """Cumulative Distribution Function (CDF) for the standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    """Probability Density Function (PDF) for the standard normal distribution."""
    return (1.0 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * x ** 2)


def _calculate_d1_d2(S, K, T, r, sigma):
    """Internal helper function to calculate d1 and d2"""
    if T <= 0 or sigma <= 0:
        return float('inf'), float('inf')
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def calculate_delta(S, K, T, r, sigma, option_type='call'):
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    if d1 == float('inf'):
        return 1.0 if option_type == 'call' else -1.0
    if option_type == 'call':
        return norm_cdf(d1)
    elif option_type == 'put':
        return norm_cdf(d1) - 1
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")


def calculate_gamma(S, K, T, r, sigma):
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    if d1 == float('inf') or T <= 0 or sigma <= 0:
        return 0.0
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def calculate_vega(S, K, T, r, sigma):
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    if d1 == float('inf') or T <= 0:
        return 0.0
    return (S * norm_pdf(d1) * math.sqrt(T)) / 100.0


def calculate_theta(S, K, T, r, sigma, option_type='call'):
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    if d1 == float('inf') or T <= 0:
        return 0.0
    if option_type == 'call':
        p1 = - (S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        p2 = r * K * math.exp(-r * T) * norm_cdf(d2)
        theta = p1 - p2
    elif option_type == 'put':
        p1 = - (S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        p2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
        theta = p1 + p2
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return theta / 365.0


def calculate_rho(S, K, T, r, sigma, option_type='call'):
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    if d1 == float('inf') or T <= 0:
        return 0.0
    if option_type == 'call':
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return rho / 100.0


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1, d2 = _calculate_d1_d2(S, K, T, r, sigma)
    if d1 == float('inf'):
        if option_type == 'call':
            return max(0.0, S - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S)
    if option_type.lower() == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type.lower() == 'put':
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return price


# --- NEW REUSABLE PLOTTING CLASS ---

class GreekPlotter:
    """
    A reusable class to create and display matplotlib plots for Streamlit.
    """

    def __init__(self, x_data, y_data, title, x_label, y_label,
                 plot_label, plot_color,
                 v_line_1_x=None, v_line_1_label=None,
                 v_line_2_x=None, v_line_2_label=None,
                 h_line_1_y=None, h_line_1_label=None,
                 invert_x=False):

        self.x_data = x_data
        self.y_data = y_data
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.plot_label = plot_label
        self.plot_color = plot_color
        self.v_line_1_x = v_line_1_x
        self.v_line_1_label = v_line_1_label
        self.v_line_2_x = v_line_2_x
        self.v_line_2_label = v_line_2_label
        self.h_line_1_y = h_line_1_y
        self.h_line_1_label = h_line_1_label
        self.invert_x = invert_x

    def display(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.x_data, self.y_data, label=self.plot_label, color=self.plot_color, linewidth=2.5)

        if self.v_line_1_x is not None:
            ax.axvline(x=self.v_line_1_x, color='gray', linestyle='--', alpha=0.7, label=self.v_line_1_label)
        if self.v_line_2_x is not None:
            ax.axvline(x=self.v_line_2_x, color='gray', linestyle=':', alpha=0.7, label=self.v_line_2_label)
        if self.h_line_1_y is not None:
            ax.axhline(y=self.h_line_1_y, color='gray', linestyle='-.', alpha=0.7, label=self.h_line_1_label)

        ax.set_title(self.title, fontsize=14)
        ax.set_xlabel(self.x_label, fontsize=12)
        ax.set_ylabel(self.y_label, fontsize=12)

        if self.invert_x:
            ax.invert_xaxis()

        if any([self.plot_label, self.v_line_1_label, self.v_line_2_label, self.h_line_1_label]):
            ax.legend(loc='best')

        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


# --- Streamlit Dashboard ---

st.set_page_config(layout="wide", page_title="Option Greeks Visualizer")
st.title("Interactive Option Greeks Dashboard")

# --- Sidebar ---
st.sidebar.header("Option Parameters")
option_type = st.sidebar.selectbox("Option Type", ('call', 'put'))
S = st.sidebar.slider("Current Stock Price ($)", 50.0, 200.0, 100.0, 0.5)
K = st.sidebar.slider("Strike Price ($)", 50.0, 200.0, 100.0, 0.5)
T = st.sidebar.slider("Time to Expiration (Years)", 0.01, 2.0, 1.0, 0.01)
r = st.sidebar.slider("Risk-Free Interest Rate (%)", 0.0, 20.0, 5.0, 0.1) / 100.0
sigma = st.sidebar.slider("Implied Volatility (%)", 1.0, 100.0, 20.0, 0.5) / 100.0

# --- Main Calculations (Single Point) ---
price = black_scholes_price(S, K, T, r, sigma, option_type)
delta = calculate_delta(S, K, T, r, sigma, option_type)
gamma = calculate_gamma(S, K, T, r, sigma)
vega = calculate_vega(S, K, T, r, sigma)
theta = calculate_theta(S, K, T, r, sigma, option_type)
rho = calculate_rho(S, K, T, r, sigma, option_type)

# --- Metrics Row ---
st.header(f"{option_type.capitalize()} Option Risk Profile")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Price", f"${price:,.2f}")
col2.metric("Delta (Δ)", f"{delta:,.3f}")
col3.metric("Gamma (Γ)", f"{gamma:,.3f}")
col4.metric("Vega (v)", f"{vega:,.3f}")
col5.metric("Theta (Θ)", f"{theta:,.3f}")
col6.metric("Rho (ρ)", f"{rho:,.3f}")

st.markdown("---")

# --- Pre-calculate standard S_range data for all tabs ---
S_range = np.linspace(S * 0.5, S * 1.5, 100)
prices_s = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]
deltas_s = [calculate_delta(s, K, T, r, sigma, option_type) for s in S_range]
gammas_s = [calculate_gamma(s, K, T, r, sigma) for s in S_range]
vegas_s = [calculate_vega(s, K, T, r, sigma) for s in S_range]
thetas_s = [calculate_theta(s, K, T, r, sigma, option_type) for s in S_range]
rhos_s = [calculate_rho(s, K, T, r, sigma, option_type) for s in S_range]

# --- TABS ---
tab_delta, tab_gamma, tab_vega, tab_theta, tab_rho = st.tabs(
    ["Δ Delta", "Γ Gamma", "v Vega", "Θ Theta", "ρ Rho"]
)

# ======================
# TAB 1: DELTA (Δ)
# ======================
with tab_delta:
    col_head, col_info = st.columns([0.8, 0.2])
    with col_head:
        st.subheader("Delta (Δ): Sensitivity to Stock Price")
    with col_info:
        with st.popover("ℹ️ Derivation"):
            st.latex(r"\Delta = \frac{\partial C}{\partial S} = N(d_1)")
            st.write("Delta is the first derivative of Price with respect to Stock Price.")

    # --- Graph 1: The Derivative (Slope) ---
    st.info("Graph 1: Delta is the **slope** of this curve (Price vs Stock Price).", icon="📈")
    GreekPlotter(S_range, prices_s, "Option Price vs. Stock Price", "Stock Price ($)", "Option Price ($)",
                 "Price", "blue", v_line_1_x=S, v_line_1_label="Current S").display()

    # --- Graph 2: The Value (Height) ---
    st.info("Graph 2: This shows the actual **value** of Delta at different stock prices.", icon="📏")
    GreekPlotter(S_range, deltas_s, "Delta vs. Stock Price", "Stock Price ($)", "Delta",
                 "Delta", "orange", v_line_1_x=S, v_line_1_label="Current S", v_line_2_x=K,
                 v_line_2_label="Strike K").display()

# ======================
# TAB 2: GAMMA (Γ)
# ======================
with tab_gamma:
    col_head, col_info = st.columns([0.8, 0.2])
    with col_head:
        st.subheader("Gamma (Γ): Sensitivity of Delta")
    with col_info:
        with st.popover("ℹ️ Derivation"):
            st.latex(
                r"\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\partial \Delta}{\partial S} = \frac{n(d_1)}{S \sigma \sqrt{T}}")
            st.write("Gamma is the derivative of Delta with respect to Stock Price.")

    # --- Graph 1: The Derivative (Slope) ---
    st.info("Graph 1: Gamma is the **slope** of this curve (Delta vs Stock Price).", icon="📈")
    GreekPlotter(S_range, deltas_s, "Delta vs. Stock Price", "Stock Price ($)", "Delta",
                 "Delta", "orange", v_line_1_x=S, v_line_1_label="Current S").display()

    # --- Graph 2: The Value (Height) ---
    st.info("Graph 2: This shows the actual **value** of Gamma. Notice it peaks AT-The-Money.", icon="📏")
    GreekPlotter(S_range, gammas_s, "Gamma vs. Stock Price", "Stock Price ($)", "Gamma",
                 "Gamma", "green", v_line_1_x=S, v_line_1_label="Current S", v_line_2_x=K,
                 v_line_2_label="Strike K").display()

# ======================
# TAB 3: VEGA (v)
# ======================
with tab_vega:
    col_head, col_info = st.columns([0.8, 0.2])
    with col_head:
        st.subheader("Vega (v): Sensitivity to Volatility")
    with col_info:
        with st.popover("ℹ️ Derivation"):
            st.latex(r"v = \frac{\partial C}{\partial \sigma} = S n(d_1) \sqrt{T}")
            st.write("Vega is the derivative of Price with respect to Volatility ($\sigma$).")

    # Data for Volatility plots
    vol_range = np.linspace(0.01, 1.0, 100)
    prices_v = [black_scholes_price(S, K, T, r, v, option_type) for v in vol_range]
    vegas_v = [calculate_vega(S, K, T, r, v) for v in vol_range]

    # --- Graph 1: The Derivative (Slope) ---
    st.info("Graph 1: Vega is the **slope** of this curve (Price vs Volatility).", icon="📈")
    GreekPlotter(vol_range, prices_v, "Option Price vs. Implied Volatility", "Volatility ($\sigma$)",
                 "Option Price ($)",
                 "Price", "blue", v_line_1_x=sigma, v_line_1_label=f"Current Vol ({sigma:.0%})").display()

    # --- Graph 2: The Value (Height) ---
    st.info("Graph 2: This shows how Vega itself changes as volatility changes.", icon="📏")
    GreekPlotter(vol_range, vegas_v, "Vega vs. Implied Volatility", "Volatility ($\sigma$)", "Vega",
                 "Vega", "red", v_line_1_x=sigma, v_line_1_label=f"Current Vol ({sigma:.0%})").display()

    # --- Graph 3: Standard View ---
    st.info("Graph 3 (Standard View): Vega at different stock prices.", icon="📊")
    GreekPlotter(S_range, vegas_s, "Vega vs. Stock Price", "Stock Price ($)", "Vega",
                 "Vega", "red", v_line_1_x=S, v_line_1_label="Current S", v_line_2_x=K,
                 v_line_2_label="Strike K").display()

# ======================
# TAB 4: THETA (Θ)
# ======================
with tab_theta:
    col_head, col_info = st.columns([0.8, 0.2])
    with col_head:
        st.subheader("Theta (Θ): Time Decay")
    with col_info:
        with st.popover("ℹ️ Derivation"):
            st.latex(r"\Theta = -\frac{\partial C}{\partial T}")
            st.write("Theta is the *negative* derivative of Price with respect to Time ($T$).")

    # Data for Time plots
    T_range_plot = np.linspace(2.0, 0.01, 100)
    prices_t = [black_scholes_price(S, K, t, r, sigma, option_type) for t in T_range_plot]
    thetas_t = [calculate_theta(S, K, t, r, sigma, option_type) for t in T_range_plot]

    # --- Graph 1: The Derivative (Slope) ---
    st.info("Graph 1: Theta is the **negative slope** of this curve. A steep drop means high time decay.", icon="📈")
    GreekPlotter(T_range_plot, prices_t, "Option Price vs. Time to Expiration", "Time to Expiration (Years)",
                 "Option Price ($)",
                 "Price Decay", "cyan", v_line_1_x=T, v_line_1_label=f"Current T ({T:.1f}y)", invert_x=True).display()

    # --- Graph 2: The Value (Height) ---
    st.info("Graph 2: This shows how Theta itself speeds up as expiration approaches.", icon="📏")
    GreekPlotter(T_range_plot, thetas_t, "Theta vs. Time to Expiration", "Time to Expiration (Years)", "Theta (Daily)",
                 "Theta", "purple", v_line_1_x=T, v_line_1_label=f"Current T ({T:.1f}y)", invert_x=True).display()

    # --- Graph 3: Standard View ---
    st.info("Graph 3 (Standard View): Theta at different stock prices.", icon="📊")
    GreekPlotter(S_range, thetas_s, "Theta vs. Stock Price", "Stock Price ($)", "Theta (Daily)",
                 "Theta", "purple", v_line_1_x=S, v_line_1_label="Current S", v_line_2_x=K,
                 v_line_2_label="Strike K").display()

# ======================
# TAB 5: RHO (ρ)
# ======================
with tab_rho:
    col_head, col_info = st.columns([0.8, 0.2])
    with col_head:
        st.subheader("Rho (ρ): Sensitivity to Interest Rates")
    with col_info:
        with st.popover("ℹ️ Derivation"):
            st.latex(r"\rho = \frac{\partial C}{\partial r}")
            st.write("Rho is the derivative of Price with respect to the Risk-Free Rate ($r$).")

    # Data for Interest Rate plots
    r_range = np.linspace(0.0, 0.20, 100)  # 0% to 20% rates
    prices_r = [black_scholes_price(S, K, T, rate, sigma, option_type) for rate in r_range]
    rhos_r = [calculate_rho(S, K, T, rate, sigma, option_type) for rate in r_range]

    # --- Graph 1: The Derivative (Slope) ---
    st.info("Graph 1: Rho is the **slope** of this curve (Price vs Interest Rate).", icon="📈")
    GreekPlotter(r_range, prices_r, "Option Price vs. Risk-Free Rate", "Risk-Free Rate ($r$)", "Option Price ($)",
                 "Price", "blue", v_line_1_x=r, v_line_1_label=f"Current Rate ({r:.0%})").display()

    # --- Graph 2: The Value (Height) ---
    st.info("Graph 2: This shows how Rho changes as interest rates change.", icon="📏")
    GreekPlotter(r_range, rhos_r, "Rho vs. Risk-Free Rate", "Risk-Free Rate ($r$)", "Rho",
                 "Rho", "magenta", v_line_1_x=r, v_line_1_label=f"Current Rate ({r:.0%})").display()

    # --- Graph 3: Standard View ---
    st.info("Graph 3 (Standard View): Rho at different stock prices.", icon="📊")
    GreekPlotter(S_range, rhos_s, "Rho vs. Stock Price", "Stock Price ($)", "Rho",
                 "Rho", "magenta", v_line_1_x=S, v_line_1_label="Current S", v_line_2_x=K,
                 v_line_2_label="Strike K").display()