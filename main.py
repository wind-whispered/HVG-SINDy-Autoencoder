# ============================================================================
# Main function
# ============================================================================
from hvg_sindy import *


def main():
    """
    Main function
    """
    # Set file path
    file_paths = {
        'Beijing': './data/Beijing.xlsx',
        'Guangdong': './data/Guangdong.xlsx',
        'Henan': './data/Henan.xlsx',
        'Shanghai': './data/Shanghai.xlsx'
    }

    # Check if the file exists
    import os
    for province, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Please check file paths.")

    # Loading data
    print("Loading data...")
    data_dict = load_data(file_paths)

    # Run comprehensive analysis
    results = comprehensive_analysis(data_dict, output_dir='results')

    print("\n" + "=" * 60)
    print("All analyses complete!")
    print("Results saved to ./results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
