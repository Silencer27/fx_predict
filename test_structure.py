"""
Test script to verify the package structure without requiring full PyTorch installation
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported (at least syntactically)"""
    print("Testing package structure...")
    
    # Test config module (without importing main package to avoid torch import)
    try:
        # Don't import through fx_predict.__init__ to avoid torch dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "fx_predict/config/config.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.get_default_config()
        print("✓ Config module working")
        print(f"  - Default config has {config['model']['num_nodes']} nodes")
    except Exception as e:
        print(f"✗ Config module failed: {e}")
        return False
    
    # Test file existence
    files_to_check = [
        'fx_predict/__init__.py',
        'fx_predict/models/gcn.py',
        'fx_predict/models/tcn.py',
        'fx_predict/models/tsgcn.py',
        'fx_predict/data/dataset.py',
        'fx_predict/data/graph_builder.py',
        'fx_predict/utils/metrics.py',
        'fx_predict/utils/visualization.py',
        'train.py',
        'evaluate.py',
        'example.py',
        'config.yaml',
        'requirements.txt',
        'setup.py',
        'README.md',
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} MISSING")
            all_exist = False
    
    return all_exist

def test_syntax():
    """Test Python syntax of all files"""
    import py_compile
    
    print("\nTesting Python syntax...")
    
    python_files = [
        'fx_predict/__init__.py',
        'fx_predict/models/__init__.py',
        'fx_predict/models/gcn.py',
        'fx_predict/models/tcn.py',
        'fx_predict/models/tsgcn.py',
        'fx_predict/data/__init__.py',
        'fx_predict/data/dataset.py',
        'fx_predict/data/graph_builder.py',
        'fx_predict/utils/__init__.py',
        'fx_predict/utils/metrics.py',
        'fx_predict/utils/visualization.py',
        'fx_predict/config/__init__.py',
        'fx_predict/config/config.py',
        'train.py',
        'evaluate.py',
        'example.py',
    ]
    
    all_valid = True
    for file_path in python_files:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✓ {file_path}")
        except py_compile.PyCompileError as e:
            print(f"✗ {file_path}: {e}")
            all_valid = False
    
    return all_valid

def main():
    print("=" * 80)
    print("TSGCN Package Structure Test")
    print("=" * 80)
    
    print("\n--- Testing File Existence ---")
    files_ok = test_imports()
    
    print("\n--- Testing Python Syntax ---")
    syntax_ok = test_syntax()
    
    print("\n" + "=" * 80)
    if files_ok and syntax_ok:
        print("✓ All tests passed!")
        print("=" * 80)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 80)
        return 1

if __name__ == '__main__':
    exit(main())
