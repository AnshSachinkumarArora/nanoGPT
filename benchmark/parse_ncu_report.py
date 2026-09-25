import sys
import ncu_report
import argparse

def extract_metrics(report_path):
    try:
        # Load the binary report file
        report = ncu_report.load_report(report_path)
    except Exception as e:
        print(f"Failed to load report: {e}")
        sys.exit(1)

    # Grab the first range and first action
    if report.num_ranges() == 0:
        print("No profiling ranges found in the report.")
        sys.exit(1)
        
    current_range = report.range_by_idx(0)
    if current_range.num_actions() == 0:
        print("No kernel actions found in the range.")
        sys.exit(1)
        
    # 'action' represents the specific kernel execution
    kernel = current_range.action_by_idx(0)
    print(f"Analyzing Kernel: {kernel.name()}\n")

    # Helper function to safely fetch metric values
    def get_metric(metric_name):
        try:
            metric = kernel.metric_by_name(metric_name)
            return metric.as_double()
        except:
            return 0.0

    # Speed of Light (SOL) Metrics
    print("--- Speed of Light & Bottlenecks ---")
    sm_pct = get_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    dram_pct = get_metric("dram__throughput.avg.pct_of_peak_sustained_elapsed")
    l1_pct = get_metric("l1tex__throughput.avg.pct_of_peak_sustained_active")
    
    print(f"Compute (SM) Throughput : {sm_pct:.2f}%")
    print(f"Global (DRAM) Throughput: {dram_pct:.2f}%")
    print(f"L1/TEX Cache Throughput : {l1_pct:.2f}%\n")

    # Occupancy & Thread Statistics
    print("--- Occupancy & Divergence ---")
    achieved_occ = get_metric("sm__warps_active.avg.pct_of_peak_sustained_active")
    registers = get_metric("launch__registers_per_thread")
    shared_mem = get_metric("launch__shared_mem_per_block_static")
    
    print(f"Achieved Occupancy      : {achieved_occ:.2f}%")
    print(f"Registers Per Thread    : {int(registers)}")
    print(f"Static Shared Mem (B)   : {int(shared_mem)}\n")

    # Roofline & Arithmetic Intensity Calculations
    print("--- Roofline Math ---")
    # Get elapsed SM cycles
    cycles = get_metric("sm__cycles_elapsed.avg")

    # FP32 Fused Multiply-Adds (1 FMA = 2 FLOPs)
    fma_instructions = get_metric("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    if fma_instructions == 0.0:
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
    parser = argparse.ArgumentParser(description="read ncu generated reports and load relevant metrics")
    parser.add_argument('--report_path', help='path to ncu report')
    args = parser.parse_args()
    extract_metrics(args.report_path)

if __name__ == "__main__":
    main()