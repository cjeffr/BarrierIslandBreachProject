#!/usr/bin/env python
"""
Script to create simplified archive datasets for figure reproduction
"""
import xarray as xr
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import yaml

def create_archive_dataset(input_file, output_file, dataset_name, description=""):
    """
    Create a simplified archive dataset with only essential variables
    """
    print(f"Processing {dataset_name}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"WARNING: Input file not found: {input_file}")
        return None
    
    try:
        # Load original dataset
        ds = xr.open_dataset(input_file)
        
        # Create new dataset with only required variables
        archive_ds = xr.Dataset(
            {
                'eta_positive': ds.eta.where(ds.eta > 0),  # Only positive eta values
                'eta_full': ds.eta,  # Keep full eta for reference
            },
            coords={
                'lat': ds.lat,
                'lon': ds.lon
            }
        )
        
        # Add metadata attributes
        archive_ds.attrs.update({
            'title': f'Storm surge data for {dataset_name}',
            'description': description,
            'source_file': os.path.basename(input_file),
            'processing_note': 'eta_positive contains only values where eta > 0, used for plotting',
            'units_eta': 'meters above mean sea level',
            'coordinate_system': 'WGS84 geographic coordinates',
            'created_for': 'Figure reproduction in manuscript'
        })
        
        # Add variable attributes
        archive_ds.eta_positive.attrs.update({
            'long_name': 'Maximum water surface elevation (positive values only)',
            'units': 'meters',
            'description': 'Used for figure plotting - equivalent to dataset.eta.where(dataset.eta > 0)'
        })
        
        archive_ds.eta_full.attrs.update({
            'long_name': 'Maximum water surface elevation (all values)',
            'units': 'meters',
            'description': 'Complete eta dataset for reference'
        })
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to archive
        archive_ds.to_netcdf(output_file, 
                            encoding={
                                'eta_positive': {'zlib': True, 'complevel': 4},
                                'eta_full': {'zlib': True, 'complevel': 4}
                            })
        
        print(f"✓ Created: {output_file}")
        ds.close()
        archive_ds.close()
        return output_file
        
    except Exception as e:
        print(f"ERROR processing {dataset_name}: {e}")
        return None

def create_figure_archive(datadir):
    """
    Create archive datasets for all figures
    """
    print("=== Creating Figure Data Archive ===")
    
    # Create output directory
    archive_dir = Path('figure_data_archive')
    archive_dir.mkdir(exist_ok=True)
    print(f"Archive directory: {archive_dir.absolute()}")
    
    # Figure 8 datasets (2x2 subplot)
    fig8_files = {
        'no_breach': {
            'file': 'no_breach_no_calc_radii_fgmax.nc',
            'description': 'Baseline scenario with no barrier island breaches'
        },
        'min_surge': {
            'file': 'dw_0283_1_fgmax.nc', 
            'description': 'Minimum surge scenario'
        },
        'max_surge': {
            'file': 'loc_0247_259_fgmax.nc',
            'description': 'Maximum surge scenario'  
        },
        'mean_surge': {
            'file': 'loc_0066_0011_fgmax.nc',
            'description': 'Mean surge scenario'
        }
    }
    
    # Figure 9 datasets (2x1 subplot with insets)
    fig9_files = {
        'scenario_w': {
            'file': 'w_0019_6_fgmax.nc',
            'description': 'Western breach scenario for detailed view'
        },
        'scenario_central': {
            'file': 'loc_0213_11_fgmax.nc', 
            'description': 'Central breach scenario for detailed view'
        }
    }
    
    # Figure 10 datasets (2x1 subplot)
    fig10_files = {
        'east_max': {
            'file': 'east_0152_91_fgmax.nc',
            'description': 'Maximum surge scenario - eastern region'
        },
        'west_max': {
            'file': 'west_0065_98_fgmax.nc',
            'description': 'Maximum surge scenario - western region'
        }
    }
    
    # Process all figures
    all_figures = {
        'figure8': fig8_files,
        'figure9': fig9_files, 
        'figure10': fig10_files
    }
    
    created_files = {}
    
    # Create datasets for each figure
    for fig_name, datasets in all_figures.items():
        print(f"\n--- Processing {fig_name.upper()} ---")
        fig_dir = archive_dir / fig_name
        fig_dir.mkdir(exist_ok=True)
        
        created_files[fig_name] = {}
        
        for dataset_name, info in datasets.items():
            input_path = os.path.join(datadir, info['file'])
            output_path = fig_dir / f"{dataset_name}.nc"
            
            result = create_archive_dataset(
                input_file=input_path,
                output_file=output_path,
                dataset_name=dataset_name,
                description=info['description']
            )
            
            if result:
                created_files[fig_name][dataset_name] = {
                    'file': str(output_path),
                    'description': info['description'],
                    'source': info['file']
                }
    
    # Create metadata file
    create_metadata_file(archive_dir, all_figures, created_files)
    
    # Create README
    create_readme_file(archive_dir)
    
    # Create reproduction script
    create_reproduction_script(archive_dir)
    
    return archive_dir

def create_metadata_file(archive_dir, all_figures, created_files):
    """
    Create a comprehensive metadata file explaining the archive structure
    """
    print("\n--- Creating Metadata ---")
    
    metadata = {
        'archive_info': {
            'title': 'Storm Surge Modeling Data Archive',
            'description': 'Simplified datasets for reproducing manuscript figures',
            'date_created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_figures': len(all_figures),
            'total_datasets': sum(len(datasets) for datasets in all_figures.values())
        },
        'coordinate_system': {
            'type': 'Geographic',
            'datum': 'WGS84',
            'lat_range': 'Approximately 40.72 to 40.83 degrees North',
            'lon_range': 'Approximately -72.88 to -72.65 degrees East'
        },
        'data_processing': {
            'eta_positive': 'Equivalent to dataset.eta.where(dataset.eta > 0) - used for figure plotting',
            'eta_full': 'Complete eta dataset for reference',
            'units': 'meters above mean sea level'
        },
        'figures': {},
        'files_created': created_files
    }
    
    for fig_name, datasets in all_figures.items():
        metadata['figures'][fig_name] = {
            'description': f'Data for {fig_name}',
            'datasets': list(datasets.keys()),
            'subplot_layout': '2x2' if fig_name == 'figure8' else '2x1',
            'dataset_count': len(datasets)
        }
    
    # Save as JSON
    json_path = archive_dir / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Created: {json_path}")
    
    # Save as YAML  
    yaml_path = archive_dir / 'metadata.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"✓ Created: {yaml_path}")

def create_readme_file(archive_dir):
    """
    Create a human-readable README file
    """
    print("--- Creating README ---")
    
    readme_content = """# Storm Surge Data Archive

This archive contains simplified datasets for reproducing figures 8, 9, and 10 from the manuscript.

## Structure

```
figure_data_archive/
├── README.md                    # This file
├── metadata.json               # Machine-readable metadata
├── metadata.yaml              # Human-readable metadata  
├── load_data_example.py       # Example usage script
├── figure8/                   # 2x2 subplot data
│   ├── no_breach.nc          # Baseline (no breaches)
│   ├── min_surge.nc          # Minimum surge scenario
│   ├── max_surge.nc          # Maximum surge scenario
│   └── mean_surge.nc         # Mean surge scenario
├── figure9/                   # 2x1 subplot with insets
│   ├── scenario_w.nc         # Western breach scenario
│   └── scenario_central.nc   # Central breach scenario
└── figure10/                  # 2x1 subplot  
    ├── east_max.nc           # Eastern maximum surge
    └── west_max.nc           # Western maximum surge
```

## Data Description

Each NetCDF file contains:
- `eta_positive`: Water surface elevation > 0 (used for plotting)
- `eta_full`: Complete water surface elevation data  
- `lat`, `lon`: Geographic coordinates (WGS84)

The `eta_positive` variable is equivalent to the original processing:
```python
dataset.eta.where(dataset.eta > 0)
```

## Usage

### Python (xarray)
```python
import xarray as xr

# Load specific dataset
ds = xr.open_dataset('figure_data_archive/figure8/no_breach.nc')
plotting_data = ds.eta_positive

# Load all figure 8 data
import glob
fig8_files = glob.glob('figure_data_archive/figure8/*.nc')
datasets = {f.split('/')[-1].replace('.nc', ''): xr.open_dataset(f) 
           for f in fig8_files}
```

### Check data
```python
# Print basic info
print(ds.eta_positive.max().values)  # Maximum water level
print(ds.eta_positive.shape)         # Grid dimensions
```

## Citation

Please cite the original manuscript when using this data.

## Contact

[Add your contact information here]
"""
    
    readme_path = archive_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created: {readme_path}")

def create_reproduction_script(archive_dir):
    """
    Create a simple script showing how to load and use the archive data
    """
    print("--- Creating Example Script ---")
    
    script_content = '''#!/usr/bin/env python
"""
Example script to load archived data and reproduce figure plotting
"""
import xarray as xr
from pathlib import Path
import glob

def load_figure_data(figure_name, dataset_name=None):
    """Load a specific dataset or all datasets for a figure"""
    archive_path = Path('figure_data_archive') / figure_name
    
    if dataset_name:
        # Load specific dataset
        file_path = archive_path / f"{dataset_name}.nc"
        if file_path.exists():
            return xr.open_dataset(file_path)
        else:
            print(f"Dataset not found: {file_path}")
            return None
    else:
        # Load all datasets for the figure
        datasets = {}
        for nc_file in archive_path.glob('*.nc'):
            dataset_name = nc_file.stem
            datasets[dataset_name] = xr.open_dataset(nc_file)
        return datasets

def print_dataset_info(ds, name):
    """Print basic information about a dataset"""
    print(f"\\n{name}:")
    print(f"  Shape: {ds.eta_positive.shape}")
    print(f"  Max elevation: {ds.eta_positive.max().values:.2f} m")
    print(f"  Non-null points: {ds.eta_positive.count().values}")

# Example usage:
if __name__ == "__main__":
    print("=== Archive Data Example ===")
    
    # Load all figure 8 data
    print("\\nLoading Figure 8 data...")
    fig8_data = load_figure_data('figure8')
    
    if fig8_data:
        for name, ds in fig8_data.items():
            print_dataset_info(ds, name)
    
    # Load specific dataset
    print("\\nLoading specific dataset...")
    no_breach = load_figure_data('figure8', 'no_breach')
    if no_breach:
        print(f"No breach scenario loaded successfully")
        print(f"Available variables: {list(no_breach.data_vars.keys())}")
        
        # This is equivalent to your original: dataset.eta.where(dataset.eta > 0)
        plotting_data = no_breach.eta_positive
        print(f"Ready for plotting: {plotting_data.shape}")
    
    # Show all available datasets
    print("\\nAvailable datasets:")
    archive_root = Path('figure_data_archive')
    for figure_dir in archive_root.iterdir():
        if figure_dir.is_dir():
            datasets = [f.stem for f in figure_dir.glob('*.nc')]
            print(f"  {figure_dir.name}: {datasets}")
'''
    
    script_path = archive_dir / 'load_data_example.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    print(f"✓ Created: {script_path}")

def main():
    """
    Main function - run this to create the archive
    """
    print("Storm Surge Data Archive Creator")
    print("=" * 40)
    
    # Get data directory from user or use default
    default_datadir = '/home/catherinej/BarrierIslandBreachProject/data/raw/fg_nc/'
    
    datadir = input(f"Enter data directory path (press Enter for default: {default_datadir}): ").strip()
    if not datadir:
        datadir = default_datadir
    
    # Check if data directory exists
    if not os.path.exists(datadir):
        print(f"ERROR: Data directory not found: {datadir}")
        print("Please check the path and try again.")
        return
    
    print(f"Using data directory: {datadir}")
    
    try:
        # Create the archive
        archive_dir = create_figure_archive(datadir)
        
        print(f"\n{'='*50}")
        print("✅ ARCHIVE CREATION COMPLETE!")
        print(f"Archive location: {archive_dir.absolute()}")
        print(f"\nFiles created:")
        
        # List all created files
        for item in sorted(archive_dir.rglob('*')):
            if item.is_file():
                rel_path = item.relative_to(archive_dir)
                print(f"  {rel_path}")
        
        print(f"\nTo use the archive:")
        print(f"1. Check README.md for detailed instructions")
        print(f"2. Run load_data_example.py to test data loading")
        print(f"3. Use the plotting script with these datasets")
        
    except Exception as e:
        print(f"ERROR: Archive creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()