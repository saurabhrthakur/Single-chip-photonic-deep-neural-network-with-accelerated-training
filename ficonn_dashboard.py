def print_performance_dashboard(final_test_accuracy):
    """
    Prints a dashboard summarizing the target FICONN hardware performance
    metrics from the paper, alongside the final simulated test accuracy.
    """
    
    # Target specifications from the paper
    target_accuracy = 92.7  # %
    latency = 435  # ps
    throughput = 0.53  # TOPS
    wavelength = 1564  # nm
    energy_per_op = 9.8  # pJ/OP

    print("\n" + "="*50)
    print("--- FICONN Performance Dashboard ---")
    print("="*50)
    
    # --- Accuracy Gauge ---
    print(f"\n[ ACCURACY ]")
    print(f"  - Paper's Target Test Accuracy : {target_accuracy:.2f}%")
    print(f"  - Our Simulated Test Accuracy  : {final_test_accuracy:.2f}%")
    # A simple text-based "gauge" for progress
    progress = int((final_test_accuracy / target_accuracy) * 20) if target_accuracy > 0 else 0
    progress = min(progress, 20) # Cap progress at 20 characters
    progress_bar = "#" * progress + "-" * (20 - progress)
    print(f"  - Progress to Target         : [{progress_bar}]")

    print("\n" + "-"*50)
    
    # --- Hardware Metric Gauges ---
    print("\n[ TARGET HARDWARE METRICS (from paper) ]\n")

    # Wavelength Gauge
    print(f"  Operating Wavelength : {wavelength} nm (Telecom C-band)")
    print(f"  [ C-BAND ]::::::::::::::::::::")

    # Latency Gauge
    print(f"  Inference Latency    : {latency} ps")
    print(f"  [ LOW LATENCY ]:::::::::::::::")
    
    # Throughput Gauge
    print(f"  Throughput           : {throughput} TOPS")
    print(f"  [ HIGH THROUGHPUT ]:::::::::")

    # Energy Efficiency Gauge
    print(f"  On-chip Energy/Op    : {energy_per_op} pJ/OP")
    print(f"  [ ENERGY EFFICIENT ]:::::::")

    print("\n" + "="*50)
