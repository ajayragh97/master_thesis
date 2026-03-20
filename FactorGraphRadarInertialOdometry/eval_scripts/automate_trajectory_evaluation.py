import json
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from evaluate_comprehensive import evaluate_trajectory  # Import our evaluation function

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG_PATH = "../config.json"
EXECUTABLE_PATH = "../build/bin/run_gt_aided_odometry"  

# Define the scenarios and their corresponding factor flags
SCENARIOS = {
    "reve_icp_imu": {"use_reve_factor": True,  "use_icp_factor": True},
    "reve_imu":     {"use_reve_factor": True,  "use_icp_factor": False},
    "icp_imu":      {"use_reve_factor": False, "use_icp_factor": True},
    "pure_imu":     {"use_reve_factor": False, "use_icp_factor": False}
}

# Define the skip rates. "blind" means NO ground truth at all after Node 0.
SKIP_RATES =[1, 10, 25, 50, 100, 250, 500, 1000, "blind"]

def run_sweep():
    # 1. Read Base Config
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    base_dir = config["dataset"]["base_dir"]
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    gt_poses_file = os.path.join(base_dir, "groundtruth", "groundtruth_poses.txt")
    gt_times_file = os.path.join(base_dir, "groundtruth", "timestamps.txt")

    all_results =[]

    # 2. Main Execution Loop
    for scenario_name, flags in SCENARIOS.items():
        print(f"\n{'='*50}\nSTARTING SCENARIO: {scenario_name}\n{'='*50}")
        
        # Create scenario specific output folder
        scenario_dir = os.path.join(results_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        for rate in SKIP_RATES:
            print(f"\n---> Running {scenario_name} | GT Skip Rate: {rate}")
            
            # Apply Ablation Flags
            config["graph"]["use_reve_factor"] = flags["use_reve_factor"]
            config["graph"]["use_icp_factor"] = flags["use_icp_factor"]

            # Apply Skip Rate / Blind Mode
            if rate == "blind":
                config["graph"]["use_gt_prior"] = False
                config["graph"]["gt_skip_rate"] = 100000 # Arbitrary high number
            else:
                config["graph"]["use_gt_prior"] = True
                config["graph"]["gt_skip_rate"] = rate

            # Set output filename so C++ saves it in the right folder
            # Note: C++ appends "_final_traj.txt" to this string
            out_filename_base = f"results/{scenario_name}/skip_{rate}"
            config["dataset"]["output_filename"] = out_filename_base
            
            # Save temporary config
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=4)

            # 3. Run C++ Executable
            try:
                # Run the process and wait for it to finish
                # stdout=subprocess.DEVNULL hides the C++ spam so we can see the Python progress
                result = subprocess.run([EXECUTABLE_PATH, CONFIG_PATH], 
                                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
                # Check if output file was actually created
                est_file = os.path.join(base_dir, out_filename_base + ".txt")
                print(f"est_file: {est_file}")
                if not os.path.exists(est_file):
                    raise Exception("Executable finished but trajectory file not found.")

                # 4. Evaluate Trajectory
                print("     Optimization successful. Evaluating trajectory...")
                metrics = evaluate_trajectory(est_file, gt_poses_file, gt_times_file, show_plots=False)
                
                if metrics:
                    res_entry = {
                        "Scenario": scenario_name,
                        "Skip_Rate": rate,
                        "Status": "SUCCESS",
                        **metrics # Unpack all RMSE values into the dict
                    }
                    all_results.append(res_entry)

            except subprocess.CalledProcessError as e:
                # Catch GTSAM Indeterminant Linear System crashes (like in pure_imu)
                print(f"[CRASH] Factor Graph failed (Likely Indeterminant System).")
                all_results.append({
                    "Scenario": scenario_name,
                    "Skip_Rate": rate,
                    "Status": "FAILED"
                })
            except Exception as e:
                print(f"     [ERROR] {str(e)}")
                all_results.append({
                    "Scenario": scenario_name,
                    "Skip_Rate": rate,
                    "Status": "FAILED"
                })

    # 5. Save Summary CSV
    df = pd.DataFrame(all_results)
    summary_path = os.path.join(results_dir, "ablation_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nAll experiments finished! Summary saved to {summary_path}")

    # 6. Generate Plot
    plot_results(df, results_dir)

def plot_results(df, results_dir):
    """ Generates a plot of 3D Positional RMSE vs Skip Rate for each scenario """
    plt.figure(figsize=(10, 6))
    
    # Filter out failures and "blind" runs for the line plot (since blind isn't a number)
    numeric_df = df[(df["Status"] == "SUCCESS") & (df["Skip_Rate"] != "blind")].copy()
    
    # Make sure skip rate is numeric
    numeric_df["Skip_Rate"] = pd.to_numeric(numeric_df["Skip_Rate"])

    markers = ['o', 's', '^', 'd']
    colors =['b', 'g', 'm', 'r']

    for i, scenario in enumerate(SCENARIOS.keys()):
        scenario_data = numeric_df[numeric_df["Scenario"] == scenario]
        if not scenario_data.empty:
            plt.plot(scenario_data["Skip_Rate"], scenario_data["rmse_pos_3d"], 
                     marker=markers[i], color=colors[i], linestyle='-', linewidth=2, label=scenario)

            # Check if "blind" succeeded for this scenario, plot it as a separate star on the far right
            blind_data = df[(df["Scenario"] == scenario) & (df["Skip_Rate"] == "blind") & (df["Status"] == "SUCCESS")]
            if not blind_data.empty:
                plt.scatter([1500], blind_data["rmse_pos_3d"], color=colors[i], marker='*', s=150)

    plt.xscale('log') # Log scale because we jump from 1 to 1000
    plt.xlabel('Ground Truth Skip Rate (Log Scale)')
    plt.ylabel('3D Absolute Trajectory Error (RMSE in meters)')
    plt.title('Odometry Degradation Analysis')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Add a note about the blind runs
    plt.text(1500, plt.gca().get_ylim()[0], 'Stars (*) represent\n"Blind" mode', horizontalalignment='center', verticalalignment='bottom')

    plot_path = os.path.join(results_dir, "ablation_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_sweep()