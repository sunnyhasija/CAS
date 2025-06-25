#!/usr/bin/env python3
"""
Simple test to verify the seeding fix works
"""

def test_seeding_fix():
    """Test that the critical bug is fixed"""
    print("🧪 Testing Seeding Fix")
    print("=" * 30)
    
    try:
        from src.scm_arena.utils.seeding import ExperimentSeeder
        
        # Test 1: Non-deterministic mode should generate different seeds
        print("\n1. Testing Non-Deterministic Mode:")
        seeder = ExperimentSeeder(base_seed=42, deterministic=False)
        
        seeds = []
        conditions = [
            ("llama3.2", "short", "specific", "local", "classic", "modern", 1),
            ("llama3.2", "short", "specific", "local", "classic", "modern", 2),
            ("llama3.2", "full", "specific", "local", "classic", "modern", 1),
        ]
        
        for condition in conditions:
            seed = seeder.get_seed(*condition)
            seeds.append(seed)
            print(f"   {condition}: {seed}")
        
        if len(set(seeds)) == len(seeds):
            print("   ✅ SUCCESS: All seeds are different!")
            print("   ✅ Non-deterministic mode is FIXED!")
        else:
            print("   ❌ FAILED: Some seeds are the same")
            return False
        
        # Test 2: Deterministic mode should be consistent
        print("\n2. Testing Deterministic Mode:")
        det_seeder = ExperimentSeeder(base_seed=42, deterministic=True)
        
        seed_a = det_seeder.get_seed("llama3.2", "short", "specific", "local", "classic", "modern", 1)
        seed_b = det_seeder.get_seed("llama3.2", "short", "specific", "local", "classic", "modern", 1)
        
        print(f"   First call:  {seed_a}")
        print(f"   Second call: {seed_b}")
        
        if seed_a == seed_b:
            print("   ✅ SUCCESS: Deterministic mode is consistent!")
        else:
            print("   ❌ FAILED: Deterministic mode inconsistent")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("🔧 The critical seeding bug has been fixed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've updated src/scm_arena/utils/seeding.py")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_seeding_fix()
    if not success:
        exit(1)