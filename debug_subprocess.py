#!/usr/bin/env python3
"""
Debug SCM-Arena Subprocess Issues
=================================

This script helps debug why experiments are starting but not completing.
"""

import subprocess
import time
import os
import threading
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_single_experiment_verbose():
    """Test running a single experiment with maximum verbosity"""
    print("🧪 Testing Single Experiment (Verbose)")
    print("=" * 50)
    
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
        "--deterministic",
        "--save-database",
        "--db-path", "test_debug.db"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nStarting experiment...")
    
    start_time = time.time()
    
    try:
        # Set environment for UTF-8
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'PYTHONLEGACYWINDOWSSTDIO': '0',
            'PYTHONUNBUFFERED': '1'  # Force unbuffered output
        })
        
        # Run with real-time output capture
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Monitor output in real-time
        def read_output(pipe, name):
            """Read output from pipe in real-time"""
            for line in iter(pipe.readline, ''):
                print(f"[{name}] {line.rstrip()}")
            pipe.close()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "OUT"))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "ERR"))
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete (with timeout)
        try:
            return_code = process.wait(timeout=120)  # 2 minute timeout
            stdout_thread.join()
            stderr_thread.join()
            
            duration = time.time() - start_time
            
            print(f"\n✅ Process completed!")
            print(f"   Return code: {return_code}")
            print(f"   Duration: {duration:.1f}s")
            
            # Check if database was created
            if os.path.exists("test_debug.db"):
                size = os.path.getsize("test_debug.db")
                print(f"   Database created: {size} bytes")
                os.remove("test_debug.db")  # Cleanup
                return True
            else:
                print(f"   ❌ No database created")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n⏰ Process timed out after 2 minutes")
            process.kill()
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        return False

def test_subprocess_hanging():
    """Test if subprocess calls are hanging"""
    print("\n🔍 Testing Basic Subprocess Calls")
    print("=" * 50)
    
    # Test 1: Simple command
    print("Test 1: Simple echo command")
    try:
        result = subprocess.run(
            ["echo", "hello"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True
        )
        print(f"   ✅ Success: {result.stdout.strip()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Poetry version
    print("Test 2: Poetry version")
    try:
        result = subprocess.run(
            ["poetry", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   ✅ Success: {result.stdout.strip()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Python in poetry env
    print("Test 3: Python in poetry environment")
    try:
        result = subprocess.run(
            ["poetry", "run", "python", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"   ✅ Success: {result.stdout.strip()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: SCM-Arena CLI help
    print("Test 4: SCM-Arena CLI help")
    try:
        result = subprocess.run(
            ["poetry", "run", "python", "-m", "scm_arena.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"   ✅ Success: CLI available")
        else:
            print(f"   ❌ Failed: Return code {result.returncode}")
            print(f"       STDERR: {result.stderr[:200]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def monitor_process_behavior():
    """Monitor what those 31 Python processes are actually doing"""
    print("\n📊 Monitoring Python Process Behavior")
    print("=" * 50)
    
    # Find all Python processes
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print(f"Found {len(python_procs)} Python processes")
    
    # Monitor for 60 seconds
    print("Monitoring CPU usage for 60 seconds...")
    for i in range(6):  # 6 x 10 seconds = 60 seconds
        total_cpu = 0
        active_count = 0
        
        for proc in python_procs[:]:  # Copy list to iterate safely
            try:
                cpu = proc.cpu_percent()
                if cpu > 0.1:  # Active process
                    active_count += 1
                total_cpu += cpu
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                python_procs.remove(proc)  # Remove dead processes
        
        print(f"   Cycle {i+1}: {active_count}/{len(python_procs)} processes active, Total CPU: {total_cpu:.1f}%")
        time.sleep(10)
    
    # Show command lines of processes
    print("\nProcess command lines:")
    for i, proc in enumerate(python_procs[:10]):  # Show first 10
        try:
            cmdline = ' '.join(proc.cmdline()[:3])  # First 3 args
            print(f"   {i+1}: {cmdline}...")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"   {i+1}: <access denied>")

def test_parallel_simple():
    """Test simple parallel execution to see if ThreadPoolExecutor is the issue"""
    print("\n🔄 Testing Simple Parallel Execution")
    print("=" * 50)
    
    def simple_task(n):
        """Simple task that should complete quickly"""
        try:
            result = subprocess.run(
                ["python", "-c", f"print('Task {n} completed')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return f"Task {n}: {result.stdout.strip()}"
        except Exception as e:
            return f"Task {n}: Failed - {e}"
    
    # Test with 4 parallel workers
    print("Running 8 simple tasks with 4 workers...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(simple_task, i) for i in range(8)]
        
        for future in as_completed(futures):
            result = future.result()
            print(f"   {result}")

def main():
    """Run all diagnostic tests"""
    print("🔍 SCM-Arena Subprocess Debug Tool")
    print("=" * 60)
    
    # Run tests
    test_subprocess_hanging()
    monitor_process_behavior()
    test_parallel_simple()
    
    # Final test - single experiment
    print("\n" + "="*60)
    success = test_single_experiment_verbose()
    
    if success:
        print("\n🎉 Single experiment test PASSED!")
        print("The issue might be with parallel execution or specific conditions.")
        print("Try reducing MAX_PARALLEL_JOBS to 2-4 in your benchmark script.")
    else:
        print("\n❌ Single experiment test FAILED!")
        print("There's a fundamental issue with running SCM-Arena experiments.")
        print("Check the output above for error messages.")
    
    print("\n💡 Recommendations:")
    print("1. If single test passed: Reduce parallel workers in benchmark")
    print("2. If processes are hanging: Check for deadlocks or infinite loops")
    print("3. If Unicode errors: Use the fixed benchmark script")
    print("4. If subprocess fails: Check Poetry environment and dependencies")

if __name__ == "__main__":
    main()