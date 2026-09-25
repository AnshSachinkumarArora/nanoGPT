import sys
import ncu_report
import argparse

def extract_metrics(kernel):        
    print(f"\n========================================")
    print(f"Analyzing Kernel: {kernel.name()}")
    print(f"========================================")

    # Helper function to safely fetch metric values
    def get_metric(metric_name):
        try:
            metric = kernel.metric_by_name(metric_name)

            num_instances = metric.num_instances()
            if num_instances > 0:
                total_samples = 0
                for idx in range(num_instances):
                    # Sample counts are collected as integers/uint64
                    total_samples += metric.as_uint64(idx)
                return float(total_samples)
            else:
                # Fallback for standard scalar metrics
                return metric.as_double()
        except Exception as e:
            print(f"[ERROR] {metric_name}: {type(e).__name__}: {e}")
            return 0.0

    # Helper function for warp stall reasons
    def get_warp_stall_reasons():
        group = kernel.metric_by_name("group:smsp__pcsamp_warp_stall_reasons")

        if group is None:
            print("[ERROR] Warp stall reason group not found")
            return {}
        
        metric_names = [name.strip() for name in group.as_string().split(",") if name.strip()]

        stall_reasons = {}

        for name in metric_names:
            metric = kernel.metric_by_name(name)
            if metric is None:
                print(f"[MISSING] {name}")
                continue
            values = [metric.as_uint64(i) for i in range(metric.num_instances())]
            stall_reasons[name] = values

        return stall_reasons

    # Speed of Light (SOL) Metrics
    print("--- Speed of Light & Bottlenecks ---")
    sm_pct = get_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem_pct = get_metric("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    l1_pct = get_metric("l1tex__throughput.avg.pct_of_peak_sustained_active")
    
    print(f"Compute (SM) Throughput : {sm_pct:.2f}%")
    print(f"GPU memory bandwidth Throughput: {mem_pct:.2f}%")
    print(f"L1/TEX Cache Throughput : {l1_pct:.2f}%\n")

    # Occupancy & Thread Statistics
    print("--- Occupancy & Divergence ---")
    achieved_occ = get_metric("sm__warps_active.avg.pct_of_peak_sustained_active")
    registers = get_metric("launch__registers_per_thread")
    shared_mem = get_metric("launch__shared_mem_per_block_static")
    
    print(f"Achieved Occupancy      : {achieved_occ:.2f}%")
    print(f"Registers Per Thread    : {int(registers)}")
    print(f"Static Shared Mem (B)   : {int(shared_mem)}\n")

    # Warp Stalling Statistics
    print("--- Warp Stalling Statistics ---")
    Memory_Dependency_Stall = get_metric("smsp__pcsamp_warps_issue_stalled_long_scoreboard")
    Execution_Pip_Busy = get_metric("smsp__pcsamp_warps_issue_stalled_math_pipe_throttle")
    Barrier_Sync_Stall = get_metric("smsp__pcsamp_warps_issue_stalled_barrier")
    Instruction_Fetch_Stall = get_metric("smsp__pcsamp_warps_issue_stalled_no_instructions")

    print(f'Memory Dependency Stall   : {Memory_Dependency_Stall} samples')
    print(f'Execution Pip Busy        : {Execution_Pip_Busy} samples')
    print(f'Barrier Sync Stall        : {Barrier_Sync_Stall} samples')
    print(f'Instruction Fetch Stall   : {Instruction_Fetch_Stall} samples\n')

    # Roofline & Arithmetic Intensity Calculations
    print("--- Roofline Math ---")
    # Get elapsed SM cycles
    cycles = get_metric("sm__cycles_elapsed.avg")

    # FP32 Fused Multiply-Adds (1 FMA = 2 FLOPs)
    fma_rate = get_metric("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed")
    fma_instructions = fma_rate * cycles
    total_flops = fma_instructions * 2

    # Global Memory Traffic (each sector is 32 bytes)
    dram_sectors_read = get_metric("dram__sectors_op_read.sum")
    if dram_sectors_read == 0.0:
        dram_sectors_read = get_metric("dram__sectors_read.sum")
        
    dram_sectors_write = get_metric("dram__sectors_op_write.sum")
    if dram_sectors_write == 0.0:
        dram_sectors_write = get_metric("dram__sectors_write.sum")

    total_bytes = (dram_sectors_read + dram_sectors_write) * 32

    print(f'FMA Rate: {fma_rate}')
    print(f'DRAM sector reads: {dram_sectors_read} DRAM sector writes: {dram_sectors_write}')
    print(f'Total bytes: {total_bytes}')
    print(f'Total flops: {total_flops}')

    # Calculate Arithmetic Intensity
    if total_bytes > 0:
        arithmetic_intensity = total_flops / total_bytes
    else:
        arithmetic_intensity = 0.0

    print(f"Total FP32 FLOPs        : {total_flops:,.0f}")
    print(f"Total DRAM Bytes        : {total_bytes:,.0f}")
    print(f"Arithmetic Intensity    : {arithmetic_intensity:.2f} FLOPs/Byte")

def main():
    # Get report path
    parser = argparse.ArgumentParser(description="read ncu generated reports and load relevant metrics")
    parser.add_argument('--report_path', help='path to ncu report')
    args = parser.parse_args()

    # Load the binary report file
    try:
        report = ncu_report.load_report(args.report_path)
    except Exception as e:
        print(f"Failed to load report: {e}")
        sys.exit(1)

    # Grab the first range as the profiler script is only capturing within a single profiler.start/stop block
    if report.num_ranges() == 0:
        print("No profiling ranges found in the report.")
        sys.exit(1)
        
    current_range = report.range_by_idx(0)
    if current_range.num_actions() == 0:
        print("No kernel actions found in the range.")
        sys.exit(1)

    for i in range(current_range.num_actions()):
        # 'action' represents the specific kernel execution
        kernel = current_range.action_by_idx(i)
        extract_metrics(kernel)

if __name__ == "__main__":
    main()