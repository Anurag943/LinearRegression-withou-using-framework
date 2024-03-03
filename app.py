import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def load_parameters(filename='model_parameters.txt', cost_filename='cost.txt'):
    with open(filename, 'r') as file:
        lines = file.readlines()
        w_str = lines[0].split(': ')[1].strip('[]\n')
        b_str = lines[1].split(': ')[1].strip('[]\n')
        w = float(w_str)
        b = float(b_str)

    # Load cost history
    cost_history = np.loadtxt(cost_filename)
    return w, b, cost_history


# Page Configuration
st.set_page_config(page_title="Linear Regression Web App", layout="wide", page_icon = '')

# Title
st.title("Linear Regression Implementation without using Frameworks")

# Sidebar for user input
num = st.number_input("Enter the study hours to predict score:",
                      min_value=0.0, max_value=9.9, step=0.5)

# Load parameters and cost history
w, b, cost_history = load_parameters()

# Prediction and Result
predicted_score = num * w + b
st.markdown("## Prediction:")
st.write(
    f"If a student studies for {num} hours, the predicted score is approximately {predicted_score:.2f}")


# Display the learned parameters
st.markdown("## Learned Parameters:")
st.write(f"- **Weight (w):** {w:.4f}")
st.write(f"- **Bias (b):** {b:.4f}")

# Display hyperparameters
st.markdown("## Hyperparameters:")
st.write("- **Learning Rate:** 0.001")
st.write("- **Iterations:** 1500")

# Display cost for every 150 iterations
st.markdown("## Cost at Selected Iterations:")
for i in range(len(cost_history)):
    if i % 150 == 0:
        st.write(f"Iteration {i}: Cost {cost_history[i]:.4f}")

# Plotting cost vs iteration
st.markdown("## Cost vs Iteration:")
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(range(len(cost_history)), cost_history,
         label='Cost', color='tab:blue')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost', color='tab:blue')
ax1.set_title('Cost vs Iteration')
ax1.legend()

# Display the plots using Streamlit
st.pyplot(fig)
