import argparse
from .synthetic import build_synthetic_dataset
from .model import summarize
from .reporting import generate_demo_artifacts


def main():
    parser = argparse.ArgumentParser(description="运行完全合成的宏观情景预测演示")
    parser.add_argument("--output-dir", default="demo_output")
    args = parser.parse_args()
    data = build_synthetic_dataset(); result = summarize(data.history, data.scenarios)
    input_path, report_path = generate_demo_artifacts(args.output_dir)
    print("宏观经济情景与经营预测 Demo（全部为合成数据）")
    print(f"历史期数：{len(data.history)}")
    print(f"模型 R2：{result['model'].r_squared:.4f}")
    print(f"滚动回测 MAPE：{result['mape']:.2%}")
    print(result["scenario_totals"].to_string(index=False))
    print(f"合成输入：{input_path.resolve()}")
    print(f"管理报告：{report_path.resolve()}")


if __name__ == "__main__": main()
