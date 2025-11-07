import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# --- Core Mathematical Functions ---
# ... (all your norm_cdf, norm_pdf, _calculate_d1_d2, etc. functions go here) ...
# ... (we'll assume they are present) ...

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
        """
        Creates and displays the matplotlib plot in Streamlit.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the main data
        ax.plot(self.x_data, self.y_data, label=self.plot_label, color=self.plot_color)

        # Add optional lines
        if self.v_line_1_x is not None:
            ax.axvline(x=self.v_line_1_x, color='r', linestyle='--', label=self.v_line_1_label)

        if self.v_line_2_x is not None:
            ax.axvline(x=self.v_line_2_x, color='g', linestyle='--', label=self.v_line_2_label)

        if self.h_line_1_y is not None:
            ax.axhline(y=self.h_line_1_y, color='b', linestyle=':', label=self.h_line_1_label)

        # Set labels and title
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        if self.invert_x:
            ax.invert_xaxis()

        # Add legend only if labels are provided
        if any([self.plot_label, self.v_line_1_label, self.v_line_2_label, self.h_line_1_label]):
            ax.legend()

        ax.grid(True)

        # Display the plot
        st.pyplot(fig)


# --- Streamlit Dashboard ---

st.set_page_config(layout="wide")
st.title("Interactive Option Greeks Dashboard")

# --- 1. Input Parameters (Sidebar) ---
st.sidebar.header("Option Parameters")

option_type = st.sidebar.selectbox("Option Type", ('call', 'put'))

S = st.sidebar.slider("Current Stock Price ($)", 50.0, 200.0, 100.0, 0.5)
K = st.sidebar.slider("Strike Price ($)", 50.0, 200.0, 100.0, 0.5)
T = st.sidebar.slider("Time to Expiration (Years)", 0.01, 2.0, 1.0, 0.01)
r = st.sidebar.slider("Risk-Free Interest Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100.0
sigma = st.sidebar.slider("Implied Volatility (%)", 1.0, 100.0, 20.0, 0.5) / 100.0

# --- 2. Calculate Greeks and Price ---
price = black_scholes_price(S, K, T, r, sigma, option_type)
delta = calculate_delta(S, K, T, r, sigma, option_type)
gamma = calculate_gamma(S, K, T, r, sigma)
vega = calculate_vega(S, K, T, r, sigma)
theta = calculate_theta(S, K, T, r, sigma, option_type)
rho = calculate_rho(S, K, T, r, sigma, option_type)

# --- 3. Display Metrics ---
st.header(f"{option_type.capitalize()} Option Risk Profile")

col1, col2, col3 = st.columns(3)
col1.metric("Option Price", f"${price:,.2f}")
col1.metric("Delta", f"{delta:,.4f}")
col2.metric("Gamma", f"{gamma:,.4f}")
col2.metric("Vega (per 1% vol)", f"${vega:,.4f}")
col3.metric("Theta (per day)", f"${theta:,.4f}")
col3.metric("Rho (per 1% rate)", f"${rho:,.4f}")

st.markdown("---")

# --- 4. Visualizations (NOW REFACTORED) ---
st.header("Greek Sensitivity Plots")

# --- Pre-calculate all data for plots vs. Stock Price ---
S_range = np.linspace(S * 0.5, S * 1.5, 100)
deltas = [calculate_delta(s, K, T, r, sigma, option_type) for s in S_range]
gammas = [calculate_gamma(s, K, T, r, sigma) for s in S_range]
vegas = [calculate_vega(s, K, T, r, sigma) for s in S_range]
thetas = [calculate_theta(s, K, T, r, sigma, option_type) for s in S_range]
prices = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]

# --- Create the tabs ---
tab_price, tab_delta, tab_gamma, tab_vega, tab_theta = st.tabs(
    ["📈 Price (vs. S)", "Δ Delta", "Γ Gamma", "v Vega", "Θ Theta"]
)

# --- Plot 1: Price ---
with tab_price:
    st.subheader("Price vs. Stock Price")

    # Use the new plotter class
    price_plot = GreekPlotter(
        x_data=S_range, y_data=prices,
        title="Price vs. Stock Price",
        x_label="Stock Price ($)", y_label="Option Price",
        plot_label="Price", plot_color='blue',
        v_line_1_x=S, v_line_1_label=f'Current S (${S:,.2f})',
        v_line_2_x=K, v_line_2_label=f'Strike K (${K:,.2f})',
        h_line_1_y=price, h_line_1_label=f'Current Price (${price:,.2f})'
    )
    price_plot.display()

# --- Plot 2: Delta ---
with tab_delta:
    # --- Title and Info Button ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("Delta vs. Stock Price")
        st.subheader("What is Delta?")
        st.write(
            "Delta (Δ) measures the rate of change of the option price with respect to a $1 change in the underlying stock price.")
        st.latex(r"\Delta = \frac{\partial C}{\partial S} = N(d_1)")
    with col2:
        with st.popover("ℹ️ Show Derivation"):
            st.subheader("Derivation Steps")
            st.write("Start with the call price formula:")
            st.latex(r"C = S N(d_1) - K e^{-rT} N(d_2)")
            st.write("Apply product and chain rules:")
            st.latex(
                r"\Delta = N(d_1) + S n(d_1) \frac{\partial d_1}{\partial S} - K e^{-rT} n(d_2) \frac{\partial d_2}{\partial S}")
            st.write(
                r"Find helpers: $\frac{\partial d_1}{\partial S} = \frac{\partial d_2}{\partial S} = \frac{1}{S \sigma \sqrt{T}}$")
            st.write("Substitute and simplify using the identity $S n(d_1) = K e^{-rT} n(d_2)$:")
            st.latex(r"\Delta = N(d_1) + \frac{n(d_1)}{\sigma \sqrt{T}} - \frac{S n(d_1)}{S \sigma \sqrt{T}}")
            st.write("Final result:")
            st.latex(r"\Delta = N(d_1)")

    # Use the new plotter class
    delta_plot = GreekPlotter(
        x_data=S_range, y_data=deltas,
        title="",  # Title is handled by st.subheader
        x_label="Stock Price ($)", y_label="Delta",
        plot_label="Delta", plot_color='orange',
        v_line_1_x=S, v_line_1_label=f'Current S (${S:,.2f})',
        v_line_2_x=K, v_line_2_label=f'Strike K (${K:,.2f})'
    )
    delta_plot.display()

# --- Plot 3: Gamma ---
with tab_gamma:
    # --- Title and Info Button ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("Gamma vs. Stock Price")
        st.subheader("What is Gamma?")
        st.write(
            "Gamma (Γ) measures the rate of change of Delta. It's the *second* partial derivative of the option price.")
        st.latex(r"\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{n(d_1)}{S \sigma \sqrt{T}}")
    with col2:
        with st.popover("ℹ️ Show Derivation"):
            st.subheader("Derivation Steps")
            st.write(r"Start with Delta: $\Delta = N(d_1)$")
            st.write(
                r"Differentiate with respect to S: $\Gamma = \frac{\partial \Delta}{\partial S} = n(d_1) \cdot \frac{\partial d_1}{\partial S}$")
            st.write(r"Substitute helper $\frac{\partial d_1}{\partial S} = \frac{1}{S \sigma \sqrt{T}}$:")
            st.latex(r"\Gamma = n(d_1) \left( \frac{1}{S \sigma \sqrt{T}} \right) = \frac{n(d_1)}{S \sigma \sqrt{T}}")

    # Use the new plotter class
    gamma_plot = GreekPlotter(
        x_data=S_range, y_data=gammas,
        title="",  # Title is handled by st.subheader
        x_label="Stock Price ($)", y_label="Gamma",
        plot_label="Gamma", plot_color='green',
        v_line_1_x=S, v_line_1_label=f'Current S (${S:,.2f})',
        v_line_2_x=K, v_line_2_label=f'Strike K (${K:,.2f})'
    )
    gamma_plot.display()

# --- Plot 4: Vega ---
with tab_vega:
    # --- Title and Info Button ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("Vega vs. Stock Price")
        st.subheader("What is Vega?")
        st.write("Vega (v) measures the option's sensitivity to a 1% change in implied volatility (σ).")
        st.latex(r"v = \frac{\partial C}{\partial \sigma} = S n(d_1) \sqrt{T}")
    with col2:
        with st.popover("ℹ️ Show Derivation"):
            st.subheader("Derivation Steps")
            st.latex(
                r"v = S n(d_1) \frac{\partial d_1}{\partial \sigma} - K e^{-rT} n(d_2) \frac{\partial d_2}{\partial \sigma}")
            st.write(r"Use identity $K e^{-rT} n(d_2) = S n(d_1)$ and factor:")
            st.latex(
                r"v = S n(d_1) \left[ \frac{\partial d_1}{\partial \sigma} - \frac{\partial d_2}{\partial \sigma} \right]")
            st.write(
                r"Use helper difference $\frac{\partial d_1}{\partial \sigma} - \frac{\partial d_2}{\partial \sigma} = \sqrt{T}$:")
            st.latex(r"v = S n(d_1) \sqrt{T}")

    # Use the new plotter class
    vega_plot = GreekPlotter(
        x_data=S_range, y_data=vegas,
        title="",  # Title is handled by st.subheader
        x_label="Stock Price ($)", y_label="Vega",
        plot_label="Vega", plot_color='red',
        v_line_1_x=S, v_line_1_label=f'Current S (${S:,.2f})',
        v_line_2_x=K, v_line_2_label=f'Strike K (${K:,.2f})'
    )
    vega_plot.display()

# --- Plot 5: Theta ---
with tab_theta:
    # --- Main Title and Info ---
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.subheader("Visualizing Theta (Θ)")
        st.write("Theta (Θ), or 'time decay', measures the option's sensitivity to the passage of time.")
    with col2:
        with st.popover("ℹ️ Show Derivation"):
            st.subheader("Derivation Steps (Call)")
            st.latex(r"\Theta = -\frac{\partial C}{\partial T}")
            st.latex(
                r"\frac{\partial C}{\partial T} = S n(d_1) \left[ \frac{\sigma}{2\sqrt{T}} \right] + r K e^{-rT} N(d_2)")
            st.write("Flip the sign for Theta:")
            st.latex(r"\Theta = -\frac{S n(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2)")

    st.markdown("---")

    # --- Nested Tabs ---

    st.info("This plot shows the option's Price vs. Time. **Theta is the *negative* of this curve's slope.**",
            icon="🧠")

    # --- Data Calculation for this Plot ---
    T_range = np.linspace(2.0, 0.01, 100)
    prices_over_time = [
        black_scholes_price(S, K, t_val, r, sigma, option_type) for t_val in T_range
    ]

    # Use the new plotter class
    theta_t_plot = GreekPlotter(
        x_data=T_range, y_data=prices_over_time,
        title="Option Price vs. Time to Expiration",
        x_label="Time to Expiration (Years)", y_label="Option Price ($)",
        plot_label="Price Decay Path", plot_color='cyan',
        v_line_1_x=T, v_line_1_label=f'Current T ({T:,.2f} yrs)',
        h_line_1_y=price, h_line_1_label=f'Current Price (${price:,.2f})',
        invert_x=True  # Invert the x-axis
    )
    theta_t_plot.display()

    st.info("This plot shows how Theta's value changes as the option approaches expiration.", icon="🧠")

    # --- 1. Calculate NEW data for this plot ---
    # Create an X-axis of Time values
    T_range = np.linspace(2.0, 0.01, 100)
    # Create a Y-axis of Theta values by looping over T_range
    thetas_over_time = [
        calculate_theta(S, K, t_val, r, sigma, option_type) for t_val in T_range
    ]

    # --- 2. Use the class with the new data ---
    theta_vs_time_plot = GreekPlotter(
        x_data=T_range, y_data=thetas_over_time,  # DATA is (Time, Theta)
        title="Theta vs. Time to Expiration",
        x_label="Time to Expiration (Years)",  # LABEL matches data
        y_label="Theta (per day)",
        plot_label="Theta", plot_color='purple',
        v_line_1_x=T, v_line_1_label=f'Current T ({T:,.2f} yrs)',  # V-LINE matches data
        invert_x=True  # This matches your other time plot
    )
    theta_vs_time_plot.display()


    st.info("This plot shows the *value* of Theta (per day) at different stock prices.", icon="⚠️")

    # Use the new plotter class
    theta_s_plot = GreekPlotter(
        x_data=S_range, y_data=thetas,
        title="Theta vs. Stock Price",
        x_label="Stock Price ($)", y_label="Theta (per day)",
        plot_label="Theta", plot_color='purple',
        v_line_1_x=S, v_line_1_label=f'Current S (${S:,.2f})',
        v_line_2_x=K, v_line_2_label=f'Strike K (${K:,.2f})'
    )
    theta_s_plot.display()

