#!/usr/bin/env python3
"""
SCM-Arena Benchmark Debug Script
===============================

This script helps debug issues with the benchmark data collector.
Run this first to identify potential problems before running the full benchmark.
"""

import subprocess
import sys
import os
import json
import sqlite3
from pathlib import Path
import psutil
import time

def check_system_requirements():
    """Check if system meets requirements"""
    print("🔍 SYSTEM REQUIREMENTS CHECK")
    print("=" * 40)
    
    # CPU cores
    cores = psutil.cpu_count()
    print(f"CPU cores: {cores} ({'✅' if cores >= 8 else '⚠️'})")
    
    # RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"RAM: {ram_gb:.1f}GB ({'✅' if ram_gb >= 32 else '⚠️'})")
    
    # GPU (if NVIDIA)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"GPU: {gpu_info} ✅")
        else:
            print("GPU: Not detected or nvidia-smi failed ⚠️")
    except:
        print("GPU: nvidia-smi not available ⚠️")
    
    print()

def check_ollama_status():
    """Check Ollama service status"""
    print("🤖 OLLAMA STATUS CHECK")
    print("=" * 40)
    
    # Check if Ollama process is running
    ollama_running = False
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            if 'ollama' in proc.info['name'].lower():
                ollama_running = True
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                print(f"Ollama process: PID {proc.info['pid']}, Memory: {memory_mb:.1f}MB ✅")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not ollama_running:
        print("Ollama process: Not running ❌")
        print("Start Ollama with: ollama serve")
        return False
    
    # Test Ollama API
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Ollama API: Responsive ✅")
            print("Available models:")
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    print(f"  - {line.split()[0]}")
        else:
            print(f"Ollama API: Error (code {result.returncode}) ❌")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Ollama API: Failed to check ({e}) ❌")
        return False
    
    print()
    return True

def check_python_environment():
    """Check Python environment and dependencies"""
    print("🐍 PYTHON ENVIRONMENT CHECK")
    print("=" * 40)
    
    # Python version
    python_version = sys.version
    print(f"Python: {python_version.split()[0]} ✅")
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("pyproject.toml: Not found ❌")
        print("Make sure you're in the SCM-Arena project directory")
        return False
    else:
        print("pyproject.toml: Found ✅")
    
    # Check Poetry
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"Poetry: {result.stdout.strip()} ✅")
        else:
            print("Poetry: Not working ❌")
            return False
    except:
        print("Poetry: Not installed ❌")
        print("Install poetry: curl -sSL https://install.python-poetry.org | python3 -")
        return False
    
    # Check if dependencies are installed
    try:
        result = subprocess.run(["poetry", "show"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("Poetry dependencies: Installed ✅")
        else:
            print("Poetry dependencies: Not installed ❌")
            print("Run: poetry install")
            return False
    except:
        print("Poetry dependencies: Could not check ❌")
        return False
    
    print()
    return True

def test_scm_arena_import():
    """Test if SCM-Arena module can be imported"""
    print("📦 SCM-ARENA MODULE CHECK")
    print("=" * 40)
    
    try:
        result = subprocess.run(
            ["poetry", "run", "python", "-c", "import scm_arena; print('Import successful')"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("SCM-Arena import: Success ✅")
        else:
            print("SCM-Arena import: Failed ❌")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"SCM-Arena import: Exception ({e}) ❌")
        return False
    
    print()
    return True

def test_model_connection_detailed(model="phi4:latest"):
    """Test model connection with detailed output"""
    print(f"🧪 MODEL CONNECTION TEST: {model}")
    print("=" * 40)
    
    # First, check if model exists in Ollama
    try:
        result = subprocess.run(["ollama", "show", model], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"Model {model}: Not found in Ollama ❌")
            print("Available models:")
            list_result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if list_result.returncode == 0:
                for line in list_result.stdout.strip().split('\n')[1:]:
                    if line.strip():
                        print(f"  - {line.split()[0]}")
            return False
        else:
            print(f"Model {model}: Found in Ollama ✅")
    except Exception as e:
        print(f"Model check failed: {e} ❌")
        return False
    
    # Test with SCM-Arena CLI
    print(f"Testing {model} with SCM-Arena CLI...")
    try:
        cmd = ["poetry", "run", "python", "-m", "scm_arena.cli", "test-model", "--model", model]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # Longer timeout
            cwd=os.getcwd()
        )
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print(f"Model {model}: SCM-Arena test passed ✅")
            return True
        else:
            print(f"Model {model}: SCM-Arena test failed ❌")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Model {model}: Test timed out ❌")
        return False
    except Exception as e:
        print(f"Model {model}: Test exception ({e}) ❌")
        return False

def test_single_experiment():
    """Test running a single experiment"""
    print("🔬 SINGLE EXPERIMENT TEST")
    print("=" * 40)
    
    # Create a minimal test experiment
    cmd = [
        "poetry", "run", "python", "-m", "scm_arena.cli", "experiment",
        "--models", "phi4:latest",
        "--memory", "none",
        "--prompts", "neutral",
        "--visibility", "local",
        "--scenarios", "classic",
        "--game-modes", "modern",
        "--runs", "1",
        "--rounds", "5",  # Very short test
        "--base-seed", "42",
        "--deterministic"
    ]
    
    print(f"Testing single experiment...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes
            cwd=os.getcwd()
        )
        duration = time.time() - start_time
        
        print(f"Duration: {duration:.1f}s")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout[-1000:]}")  # Last 1000 chars
        if result.stderr:
            print(f"STDERR:\n{result.stderr[-1000:]}")  # Last 1000 chars
        
        if result.returncode == 0:
            print("Single experiment: Success ✅")
            return True
        else:
            print("Single experiment: Failed ❌")
            return False
            
    except subprocess.TimeoutExpired:
        print("Single experiment: Timed out ❌")
        return False
    except Exception as e:
        print(f"Single experiment: Exception ({e}) ❌")
        return False

def check_file_permissions():
    """Check file permissions for output files"""
    print("📁 FILE PERMISSIONS CHECK")
    print("=" * 40)
    
    # Check current directory permissions
    current_dir = Path.cwd()
    if os.access(current_dir, os.W_OK):
        print(f"Current directory: Writable ✅")
    else:
        print(f"Current directory: Not writable ❌")
        return False
    
    # Check if we can create test files
    test_files = ["test_db.db", "test_progress.json"]
    for test_file in test_files:
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"Test file {test_file}: Can create/delete ✅")
        except Exception as e:
            print(f"Test file {test_file}: Cannot create ({e}) ❌")
            return False
    
    print()
    return True

def main():
    """Run all diagnostic checks"""
    print("🔍 SCM-Arena Benchmark Diagnostic Tool")
    print("=" * 50)
    print()
    
    checks = [
        ("System Requirements", check_system_requirements),
        ("Python Environment", check_python_environment),
        ("File Permissions", check_file_permissions),
        ("Ollama Status", check_ollama_status),
        ("SCM-Arena Module", test_scm_arena_import),
        ("Model Connection", lambda: test_model_connection_detailed("phi4:latest")),
        ("Single Experiment", test_single_experiment)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"Running {check_name} check...")
        try:
            success = check_func()
            results[check_name] = success
            if success:
                print(f"{check_name}: ✅ PASSED")
            else:
                print(f"{check_name}: ❌ FAILED")
        except Exception as e:
            print(f"{check_name}: ❌ EXCEPTION: {e}")
            results[check_name] = False
        
        print("-" * 30)
        print()
    
    # Summary
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    passed = sum(results.values())
    total = len(results)
    
    for check_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your system should be ready to run the benchmark.")
    else:
        print(f"\n⚠️ {total - passed} checks failed. Fix these issues before running the benchmark.")
        
        # Specific recommendations
        if not results.get("Ollama Status", True):
            print("\n💡 OLLAMA ISSUES:")
            print("  1. Start Ollama: ollama serve")
            print("  2. Pull the model: ollama pull phi4:latest")
            print("  3. Verify: ollama list")
        
        if not results.get("Python Environment", True):
            print("\n💡 PYTHON ENVIRONMENT ISSUES:")
            print("  1. Install Poetry: curl -sSL https://install.python-poetry.org | python3 -")
            print("  2. Install dependencies: poetry install")
            print("  3. Verify: poetry show")
        
        if not results.get("SCM-Arena Module", True):
            print("\n💡 SCM-ARENA MODULE ISSUES:")
            print("  1. Make sure you're in the project directory")
            print("  2. Install dependencies: poetry install")
            print("  3. Check project structure")

if __name__ == "__main__":
    main()