# ============================================================================
# Main function
# ============================================================================
from hvg_sindy import load_data, comprehensive_analysis
from comparison import run_comparison, plot_comparison
from statsmodels.tools.sm_exceptions import ConvergenceWarning


import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

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
    # results = comprehensive_analysis(data_dict, output_dir='results')
    results, processed_data, sindy_models, \
        province_latents, autoencoder = \
        comprehensive_analysis(data_dict, output_dir='results')

    # Comparative analysis
    print("\n" + "=" * 60)
    print("Running comparative analysis...")
    print("=" * 60)

    comparison_results = run_comparison(
        processed_data=processed_data,
        sindy_models=sindy_models,
        province_latents=province_latents,
        autoencoder=autoencoder,
        all_labels=None,
        output_dir='results',
        test_len=50
    )

    plot_comparison(comparison_results, output_dir='results', test_len=50)

    print("\n" + "=" * 60)
    print("All analyses complete!")
    print("Results saved to ./results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
