#!/bin/bash

# --- Configuration ---
RUNTIME_HOURS=6       # Restart interval in hours
RUNTIME_SECONDS=$((RUNTIME_HOURS * 3600))  # Convert hours to seconds

if [ "$HOSTNAME" = "penguin" ]; then
    echo "Running on $HOSTNAME"
    export MPLBACKEND=Agg
    CONDA_ENV="WFP2"
fi
if [ "$HOSTNAME" = "GLaDOS" ]; then
    echo "Running on $HOSTNAME"
    CONDA_ENV="WFP-G2"
fi

# --- Functions ---

# Function to start or restart a model
run_model() {
    local model_name="$1"
    # Use underscores consistently for the session name
    local session_name=$(echo "$model_name" | sed 's/\.py/_py/')


    # Check if the tmux session already exists
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Session '$session_name' already exists.  Killing..."
        tmux kill-session -t "$session_name"
    fi

    echo "Starting '$model_name' in tmux session '$session_name'..."

    # Create a new detached tmux session
    tmux new-session -d -s "$session_name"

    # Get the current TMUX variable (if it exists)
    local current_tmux_var=$(tmux show-environment TMUX 2>/dev/null | cut -d '=' -f 2-)

    # Activate the conda environment (and set environment variables)
    if [[ -n "$current_tmux_var" ]]; then
        tmux send-keys -t "$session_name" "export TMUX=$current_tmux_var" C-m
    fi

    tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" C-m
    # Corrected line: Use session_name, not model_name
    #tmux send-keys -t "$session_name" "while true; do timeout ${RUNTIME_SECONDS}s python3 $model_name; sleep 5; done" C-m
    tmux send-keys -t "$session_name" "while true; do timeout ${RUNTIME_SECONDS}s python3 -m src.ThesisProject.LSTM.$model_name; sleep 5; done" C-m

    echo "Started '$model_name'.  Will restart every $RUNTIME_HOURS hours."
}

# --- Main Script Execution ---

# Array of model names
models=("model3_0.py" "model3_1.py" "model3_2.py" "model3_3.py" "model3_4.py" "model3_5.py" "model3_6.py" "model3_7.py" "model3_8.py" "model3_9.py" "model3_10.py" "model3_11.py")


# Loop through each model and start/restart it
for model in "${models[@]}"; do
    run_model "$model"
done

echo "All models started.  Use 'tmux a -t <session_name>' (e.g., tmux a -t model3_py) to attach to a session."

exit 0
